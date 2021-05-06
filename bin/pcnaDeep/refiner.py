# -*- coding: utf-8 -*-
import math
import logging
import warnings
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.svm import SVC
import skimage.morphology as morph
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from pcnaDeep.data.utils import get_outlier
from imblearn.over_sampling import BorderlineSMOTE


def dist(x1, y1, x2, y2):
    """Calculate distance of a set of coordinates
    """
    return math.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2)


def deduce_transition(l, tar, confidence, min_tar, max_res, escape=0, casual_end=True):
    """ Deduce mitosis exit and entry based on adaptive searching
        
        Args:
            l (list): list of the target cell cycle phase
            tar (str): target cell cycle phase
            min_tar (int): minimum duration of an entire target phase
            confidence (numpy.array): matrix of confidence
            max_res (int): maximum accumulative duration of unwanted phase
            escape (int): do not consider the first n instances
            casual_end (bool): at the end of the track, whether loosen criteria of a match
            
        Returns:
            (entry, exit), index of index list that resolved as entry and exit
        """
    mp = {'G1/G2': 0, 'S': 1, 'M': 2}
    confid_cls = list(map(lambda x: confidence[x, mp[l[x]]], range(confidence.shape[0])))
    idx = np.where(np.array(l) == tar)[0]
    idx = idx[idx >= escape].tolist()
    if len(idx) == 0:
        return None
    if len(idx) == 1:
        return idx[0], idx[0]
    found = False
    i = 0
    g_panelty = 0
    acc_m = confid_cls[idx[0]]
    cur_m_entry = idx[i]
    m_exit = None
    while i < len(idx) - 1:
        acc_m += confid_cls[idx[i + 1]]
        g_panelty += np.sum(confid_cls[idx[i] + 1:idx[i + 1]])
        if acc_m >= min_tar:
            found = True
        if g_panelty >= max_res:
            if found:
                m_exit = idx[i]
                break
            else:
                g_panelty = 0
                acc_m = 0
                cur_m_entry = idx[i + 1]
        i += 1
    if i == (len(idx) - 1) and found:
        m_exit = idx[-1]
    elif casual_end and i == (len(idx) - 1) and g_panelty < max_res and not found and cur_m_entry != idx[-1]:
        found = True
        m_exit = idx[-1]
        if m_exit - cur_m_entry + 1 < min_tar:
            return None

    if found and m_exit is not None:
        return cur_m_entry, m_exit
    else:
        return None


class Refiner:

    def __init__(self, track, smooth=5, minGS=6, minM=3, mode='SVM',
                 threshold_mt_F=150, threshold_mt_T=5,
                 search_range=5, mt_len=5, sample_freq=1/20, model_train='', mask=None):
        """Refinement of the tracks

        Class variables:
            track (pandas.DataFrame): tracked object table
            smooth (int): smoothing window on classification confidence
            minGS (int): minimum duration of G1/G2/S phase, choose maximum if differs among three
            mode (str): how to resolve parent-daughte relationship, either 'SVM', 'TRAIN' or 'TRH'
            - Essential for TRH mode:
                threshold_mt_F (int): mitosis displace maximum, can be evaluated as maximum cytokinesis distance.
                threshold_mt_T (int): mitosis frame difference maximum,
                    can be evaluated as maximum mitosis frame length.
            - Essential for SVM/TRAIN mode (for normalizing different imaging conditions):
                search_range (int): when calculating mitosis score, how many time points to consider
                mt_len (int): mitosis length of the cells, evaluated manually
                sample_freq (int): sampling frequency: x minute per frame
                model_train (str): path to SVM model training data
            mask (numpy.array): object masks, same shape as input, must labeled with object ID
        """
        self.logger = logging.getLogger('pcna.Refiner')
        self.flag = False
        self.mask = mask
        self.track = track.copy()
        self.count = np.unique(track['trackId'])

        self.MODE = mode
        self.MT_DISCOUNT = 0.9
        self.metaData = {'mt_len': mt_len, 'sample_freq': sample_freq,
                         'meanDisplace': np.mean(self.getMeanDisplace()['mean_displace'])}
        self.logger.info(self.metaData)
        self.SEARCH_RANGE = search_range
        self.mt_score_begin, self.mt_score_end = self.getMTscore(self.SEARCH_RANGE, self.MT_DISCOUNT)
        if mode == 'SVM' or mode == 'TRAIN':
            self.SVM_PATH = model_train
        elif mode == 'TRH':
            self.FRAME_MT_TOLERANCE = threshold_mt_T
            self.DIST_MT_TOLERANCE = threshold_mt_F
        else:
            raise ValueError('Mode can only be SVM, TRAIN or TRH, for SVM-mitosis resolver, '
                             'training of the resolver or threshold based resolver.')

        self.SMOOTH = smooth
        self.MIN_GS = minGS
        self.MIN_M = minM
        self.short_tracks = []
        self.daug_from_broken = []
        self.mt_dic = {}
        self.par_mt_mask = {}
        self.mt_exit_lookup = {}  # {parent ID: (exit frame, quality)}
        self.mt_entry_lookup = {}  # {daughter ID: (exit frame, quality)}
        self.imprecise = []  # imprecise mitosis: daughter exit without M classification
        self.mean_size = np.mean(np.array(self.track[['major_axis', 'minor_axis']]))
        self.mean_intensity = np.mean(np.array(self.track[['mean_intensity']]))
        self.ann = pd.DataFrame(
            columns=['track', 'app_frame', 'disapp_frame', 'app_x', 'app_y', 'disapp_x', 'disapp_y', 'app_stage',
                     'disapp_stage', 'predicted_parent'])

    def break_mitosis(self):
        """break mitosis tracks
        iterate until no track is broken
        """
        track = self.track.copy()
        cur_max = np.max(track['trackId']) + 1
        count = 0
        track = track.sort_values(by=['trackId', 'frame'])
        filtered_track = pd.DataFrame(columns=track.columns)
        for trk in list(np.unique(track['trackId'])):
            sub = track[track['trackId'] == trk].copy()
            if trk in list(self.mt_dic.keys()):
                filtered_track = filtered_track.append(sub.copy())
                continue

            found = False
            if sub.shape[0] > self.MIN_GS and 'M' in list(sub['predicted_class']):
                cls = sub['predicted_class'].tolist()
                confid = np.array(sub[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
                if sub['parentTrackId'].iloc[0] in self.mt_dic.keys():
                    prev_exit = self.mt_dic[int(sub['parentTrackId'].iloc[0])]['daug'][trk]['m_exit']
                    esp = list(sub['frame']).index(prev_exit)+1
                else:
                    esp = self.MIN_M
                out = deduce_transition(l=cls, tar='M', confidence=confid, min_tar=self.MIN_M, max_res=self.MIN_GS,
                                        escape=esp)

                if out is not None and out[0] != out[1] and out[1] != len(cls) - 1:
                    found = True
                    cur_m_entry, m_exit = out
                    cla = list(sub['predicted_class'])
                    for k in range(cur_m_entry, m_exit+1):
                        cla[k] = 'M'
                    
                    sub.loc[:, 'predicted_class'] = cla
                    # split mitosis track, keep parent track with 2 'M' prediction
                    # this makes cytokinesis unpredictable...
                    m_entry = list(sub['frame'])[cur_m_entry]
                    x_list = list(sub['Center_of_the_object_0'].iloc[cur_m_entry:m_exit+1])
                    y_list = list(sub['Center_of_the_object_1'].iloc[cur_m_entry:m_exit+1])
                    frame_list = list(sub['frame'])
                    distance = \
                        list(map(lambda x:dist(x_list[x], y_list[x],
                                               x_list[x+1], y_list[x+1]) / (frame_list[x+1]-frame_list[x]),
                                 range(len(x_list)-1)))
                    sp_time = cur_m_entry + np.argmax(distance)+1
                    new_track = sub[sub['frame'] >= frame_list[sp_time]].copy()
                    new_track.loc[:, 'trackId'] = cur_max
                    new_track.loc[:, 'lineageId'] = list(sub['lineageId'])[0]  # inherit the lineage
                    new_track.loc[:, 'parentTrackId'] = trk  # mitosis parent asigned
                    # register to the class
                    x1 = sub.iloc[sp_time]['Center_of_the_object_0']
                    y1 = sub.iloc[sp_time]['Center_of_the_object_1']
                    x2 = sub.iloc[sp_time - 1]['Center_of_the_object_0']
                    y2 = sub.iloc[sp_time - 1]['Center_of_the_object_1']
                    self.mt_dic[trk] = {'div': m_entry,
                                        'daug': {cur_max: {'m_exit': frame_list[m_exit], 
                                            'dist': np.round(dist(x1, y1, x2, y2), 2)}}}
                    cur_max += 1
                    count += 1
                    old_track = sub[sub['frame'] < frame_list[sp_time]].copy()
                    filtered_track = filtered_track.append(old_track.copy())
                    filtered_track = filtered_track.append(new_track.copy())
            if not found:
                filtered_track = filtered_track.append(sub.copy())

        self.logger.info('Found mitosis track: ' + str(count))
        return filtered_track, count

    def register_track(self):
        """Register track annotation table 
        """
        if self.MODE == 'TRH':
            frame_tolerance = self.FRAME_MT_TOLERANCE
        else:
            frame_tolerance = self.SEARCH_RANGE

        track = self.track.copy()
        # annotation table: record appearance and disappearance information of the track
        track_count = len(np.unique(track['trackId']))
        ann = {"track": [i for i in range(track_count)],
               "app_frame": [0 for _ in range(track_count)],
               "disapp_frame": [0 for _ in range(track_count)],
               "app_x": [0 for _ in range(track_count)],  # appearance coordinate
               "app_y": [0 for _ in range(track_count)],
               "disapp_x": [0 for _ in range(track_count)],  # disappearance coordinate
               "disapp_y": [0 for _ in range(track_count)],
               "app_stage": [None for _ in range(track_count)],  # cell cycle classification at appearance
               "disapp_stage": [None for _ in range(track_count)],  # cell cycle classification at disappearance
               # "predicted_parent" : [None for _ in range(track_count)], # non-mitotic parent track TO-predict
               # "predicted_daughter" : [None for _ in range(track_count)],
               "mitosis_parent": [None for _ in range(track_count)],  # mitotic parent track to predict
               "mitosis_daughter": ['' for _ in range(track_count)],
               "m_entry": [None for _ in range(track_count)],
               "m_exit": [None for _ in range(track_count)],
               "mitosis_identity": ['' for _ in range(track_count)]
               }

        short_tracks = []
        trks = list(np.unique(track['trackId']))
        for i in range(track_count):
            cur_track = track[track['trackId'] == trks[i]]
            # constraint A: track < 2 frame length tolerance is filtered out, No relationship can be deduced from that.
            ann['track'][i] = trks[i]
            # (dis-)appearance time
            ann['app_frame'][i] = min(cur_track['frame'])
            ann['disapp_frame'][i] = max(cur_track['frame'])
            # (dis-)appearance coordinate
            ann['app_x'][i] = cur_track['Center_of_the_object_0'].iloc[0]
            ann['app_y'][i] = cur_track['Center_of_the_object_1'].iloc[0]
            ann['disapp_x'][i] = cur_track['Center_of_the_object_0'].iloc[cur_track.shape[0] - 1]
            ann['disapp_y'][i] = cur_track['Center_of_the_object_1'].iloc[cur_track.shape[0] - 1]
            rt = self.render_emerging(track=cur_track, cov_range=frame_tolerance)
            ann['app_stage'][i] = rt[0]
            ann['disapp_stage'][i] = rt[1]

            if max(cur_track['frame']) - min(cur_track['frame']) < frame_tolerance:
                short_tracks.append(trks[i])
        self.short_tracks = short_tracks.copy()

        ann = pd.DataFrame(ann)
        # register mitosis relationship from break_mitosis()
        for i in list(self.mt_dic.keys()):
            daug_trk = list(self.mt_dic[i]['daug'].keys())[0]
            # parent
            idx = ann[ann['track'] == i].index
            ann.loc[idx, 'mitosis_identity'] = ann.loc[idx, 'mitosis_identity'] + '/' + 'parent'
            ann.loc[idx, 'mitosis_daughter'] = daug_trk
            ann.loc[idx, 'm_entry'] = self.mt_dic[i]['div']

            # daughter
            idx = ann[ann['track'] == daug_trk].index
            ann.loc[idx, 'mitosis_identity'] = ann.loc[idx, 'mitosis_identity'] + '/' + 'daughter'
            ann.loc[idx, 'mitosis_parent'] = i
            ann.loc[idx, 'm_exit'] = self.mt_dic[i]['daug'][daug_trk]['m_exit']

        track['lineageId'] = track['trackId'].copy()  # erase original lineage ID, assign in following steps
        self.logger.info("High quality tracks subjected to predict relationship: " + str(ann.shape[0] - len(short_tracks)))

        return track, short_tracks, ann

    def render_emerging(self, track, cov_range):
        """Render emerging phase
        """
        if track.shape[0] >= 2 * cov_range:
            bg_cls = list(track['predicted_class'].iloc[0:cov_range])
            bg_emg = list(track['emerging'].iloc[0:cov_range])
            end_cls = list(track['predicted_class'].iloc[(track.shape[0] - cov_range): track.shape[0]])
            end_emg = list(track['emerging'].iloc[(track.shape[0] - cov_range): track.shape[0]])
            
        else:
            bg_cls = list(track['predicted_class'].iloc[0:min(cov_range, track.shape[0])])
            bg_emg = list(track['emerging'].iloc[0:min(cov_range, track.shape[0])])
            end_cls = list(track['predicted_class'].iloc[max(0, track.shape[0] - cov_range):])
            end_emg = list(track['emerging'].iloc[max(0, track.shape[0] - cov_range):])
        
        for i in range(len(bg_emg)):
            if bg_emg[i] == 1:
                bg_cls[i] = 'M'
        for i in range(len(end_cls)):
            if end_emg[i] == 1:
                end_cls[i] = 'M'
        return '-'.join(bg_cls), '-'.join(end_cls)

    def revert(self, ann, mt_dic, parentId, daughterId):
        """Remove information of a relationship registered to ann and mt_dic
        """
        self.logger.info('Revert: ' + str(parentId) + '-' + str(daughterId))
        # parent
        mt_dic[parentId]['daug'].pop(daughterId)
        ori = ann.loc[ann['track'] == parentId]['mitosis_identity'].values[0]
        ori = ori.replace('/parent', '', 1)
        ann.loc[ann['track'] == parentId, 'mitosis_identity'] = ori
        ori_daug = str(ann.loc[ann['track'] == parentId]['mitosis_daughter'].values[0])
        ori_daug = ori_daug.split('/')
        ori_daug.remove(str(daughterId))
        ann.loc[ann['track'] == parentId, 'mitosis_daughter'] = '/'.join(ori_daug)
        # daughter
        ori = ann.loc[ann['track'] == daughterId]['mitosis_identity'].values[0]
        ori = ori.replace('/daughter', '', 1)
        ann.loc[ann['track'] == daughterId, 'mitosis_identity'] = ori
        ann.loc[ann['track'] == daughterId, 'm_exit'] = None
        ann.loc[ann['track'] == daughterId, 'mitosis_parent'] = None

        return ann, mt_dic

    def register_mitosis(self, ann, mt_dic, parentId, daughterId, m_exit, dist_dif, m_entry=0):
        """Register parent and dduahgter information to ann and mt_dic
        """
        self.logger.info('Register: ' + str(parentId) + '-' + str(daughterId))
        ori = ann.loc[ann['track'] == parentId, "mitosis_daughter"].values[0]
        ori_idt = ann.loc[ann['track'] == parentId, "mitosis_identity"].values[0]
        s1 = ann.loc[ann['track'] == daughterId, "mitosis_identity"].values
        ann.loc[ann['track'] == daughterId, "mitosis_identity"] = s1 + "/daughter"
        ann.loc[ann['track'] == daughterId, "m_exit"] = m_exit
        ann.loc[ann['track'] == daughterId, "mitosis_parent"] = parentId
        ann.loc[ann['track'] == parentId, "mitosis_daughter"] = str(ori) + '/' + str(daughterId)
        ann.loc[ann['track'] == parentId, "mitosis_identity"] = str(ori_idt) + '/parent'
        if parentId not in list(mt_dic.keys()):
            mt_dic[parentId] = {'div': m_entry, 'daug': {}}
        mt_dic[parentId]['daug'][daughterId] = {'m_exit': m_exit, 'dist': dist_dif}
        return ann, mt_dic

    def getMtransition(self, trackId, direction='entry', skip=0):
        """Get mitosis transition time by trackId

        Args:
            trackId (int): track ID
            direction (str): either 'entry' or 'exit', mitosis entry or exit
            skip (int): escape frames from deduce_transition() method
        """
        skp = None
        trk = self.track[self.track['trackId'] == trackId].copy()
        c1 = list(trk['predicted_class'])
        c1_confid = np.array(trk[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
        if direction == 'exit':
            daug_entry = self.ann[self.ann['track'] == trackId]['m_entry'].values[0]
            if daug_entry is not None:
                skp = list(trk['frame']).index(daug_entry)
                c1 = c1[:skp]
                c1_confid = c1_confid[:skp, :]
            trans = deduce_transition(c1, tar='M', confidence=c1_confid, min_tar=1, max_res=self.MIN_GS, escape=skip)
            if trans is not None:
                trans = trans[1]
            else:
                return None
        elif direction == 'entry':
            par_exit = self.ann[self.ann['track'] == trackId]['m_exit'].values[0]
            if par_exit is not None:
                skp = list(trk['frame']).index(par_exit)
                c1 = c1[skp+1:]
                c1_confid = c1_confid[skp+1:, :]
            trans = deduce_transition(c1[::-1], tar='M', confidence=c1_confid[::-1, :],
                                      min_tar=1, max_res=self.MIN_GS, escape=skip)
            if trans is not None:
                trans = len(c1) - (1 + trans[1])
            else:
                return None
        else:
            raise ValueError('Direction can either be entry or exit')

        if skp is not None and direction == 'entry':
            trans = list(trk['frame'])[trans + skp + 1]
        else:
            trans = list(trk['frame'])[trans]

        return trans

    def extract_pools(self, extra_par=None, extra_daug=None):
        """Extract potential parent and daughter pool
        """
        daughter_pool = []
        parent_pool = list(self.mt_dic.keys())
        for i in self.mt_dic.keys():
            for j in self.mt_dic[i]['daug'].keys():
                daughter_pool.append(j)
        self.daug_from_broken = daughter_pool.copy()
        pool = list(np.unique(self.track['trackId']))
        lin_par_pool = list(set(np.unique(self.track['parentTrackId'])) - set(parent_pool))
        lin_daug_pool = list(set(np.unique(self.track[self.track['parentTrackId'] > 0]['trackId'])) -
                             set(daughter_pool))

        for i in pool:
            if i not in parent_pool and i not in lin_par_pool and i not in self.short_tracks:
                # wild parents: at least two M classification at the end
                if re.search('M', self.ann[self.ann['track'] == i]['disapp_stage'].values[0]) is not None:
                    parent_pool.append(i)

        for i in pool:
            if i not in daughter_pool and i not in lin_daug_pool and i not in self.short_tracks:
                if re.search('M', self.ann[self.ann['track'] == i]['app_stage'].values[0]) is not None:
                    daughter_pool.append(i)

        if extra_par:
            parent_pool = list(set(parent_pool) | set(extra_par))
        if extra_daug:
            daughter_pool = list(set(daughter_pool) | set(extra_daug))

        return parent_pool, daughter_pool

    def get_parent_mask(self, p):
        """Extract parent mask, begin from mitosis entry, end with parent disappearance

        Args:
            p (int): parent track ID
        """
        if p not in self.par_mt_mask.keys():
            sub = self.track[(self.track['trackId'] == p) & (self.track['frame'] >= (self.mt_entry_lookup[p][0] - 2))]
            lbs = list(sub['continuous_label'])
            frame = list(sub['frame'])
            sls = []
            for i in range(len(lbs)):
                sl = self.mask[frame[i], :, :].copy()
                sl[sl != lbs[i]] = 0
                sl = sl.astype('bool')
                sls.append(sl)
            out = np.sum(np.stack(sls, axis=0), axis=0)
            out = out.astype('bool')
            dilate_range = int(np.floor(self.mean_size/4))  # dilate the mask by 50% mean radius
            out = morph.binary_dilation(out, selem=np.ones((dilate_range, dilate_range)))
            if np.sum(out) == 0:
                warnings.warn('Object not found in mask for parent: ' + str(p) + ' in frames: ' + str(frame)[1:-1])
            self.par_mt_mask[p] = out
            '''
            import skimage.io as io
            io.imsave('../../test/test_files/mask/'+str(p)+'.tif', out)
            '''
            return out
        else:
            return self.par_mt_mask[p]

    def daug_app_in_par_mask(self, par, daug):
        """Check if daughter appears in the mask of parent

        Args:
            par (int): parent track ID
            daug (int): daughter track ID
        """
        mask = self.get_parent_mask(par)
        sub = self.ann[self.ann['track'] == daug]
        x = int(np.floor(sub['app_x']))
        y = int(np.floor(sub['app_y']))
        if mask[y, x]:
            return True
        else:
            return False

    def extract_features(self, par_pool, daug_pool, remove_outlier=None, normalize=None, sample=None):
        """Extract Input Features for the classifier

        Args:
            par_pool (list): parent pool
            daug_pool (list): daughter pool
            remove_outlier ([int]): remove outlier of columns in the feature map
            normalize (bool): normalize each column
            sample (numpy.array): training mode only, supply positive sample information, will add y as 2nd output
        """
        if sample is not None and self.MODE != 'TRAIN':
            raise NotImplementedError('Only allowed to input sample in TRAIN mode.')

        self.logger.info('Extracting features...')
        ft = 0
        ipts = []
        sample_id = []
        y = []
        for i in par_pool:
            for j in range(len(daug_pool)):
                if i != daug_pool[j]:
                    if ft % 500 == 0 and ft > 0:
                        self.logger.info('Considered ' + str(ft) + '/' + str(len(daug_pool) * len(par_pool)) + ' cases.')
                    ft += 1
                    ind = self.getSVMinput(i, daug_pool[j])

                    rgd = False  # first register input from broken pairs
                    if i in self.mt_dic.keys():
                        if daug_pool[j] in self.mt_dic[i]['daug'].keys():
                            ipts.append(ind)
                            sample_id.append([i, daug_pool[j]])
                            rgd = True

                    if not rgd and (ind[0] >= 3 or ind[1] <= 0 or
                                    ind[1] > 3 * self.metaData['mt_len'] / self.metaData['sample_freq']):
                        # if distance over 3 (average radius + average move * time frame difference), discard it.
                        continue
                    elif not rgd:
                        if self.mask is not None:
                            if not self.daug_app_in_par_mask(i, daug_pool[j]):
                                continue
                        ipts.append(ind)
                        sample_id.append([i, daug_pool[j]])

                    if sample is not None:
                        a = np.where(sample[:, 0] == i)[0].tolist()
                        b = np.where(sample[:, 1] == daug_pool[j])[0].tolist()
                        sp_index = list(set(a) & set(b))
                        if sp_index:
                            y.append(1)
                        else:
                            y.append(0)
        ipts = np.array(ipts)
        sample_id = np.array(sample_id)
        if sample is not None:
            y = np.array(y)

        if remove_outlier is not None:
            outs = get_outlier(ipts, col_ids=remove_outlier)
            idx = [_ for _ in range(ipts.shape[0]) if _ not in outs]
            ipts = ipts[idx,]
            sample_id = sample_id[idx,]
            if sample is not None:
                y = y[idx,]
            self.logger.info('Removed outliers, remaining: ' + str(ipts.shape[0]))

        if normalize:
            scaler = RobustScaler()
            ipts = scaler.fit_transform(ipts)

        self.logger.info('Finished feature extraction: ' + str(ipts.shape[0]) + ' samples.')
        if sample is not None:
            return ipts, y, sample_id
        else:
            return ipts, sample_id

    def plainPredict(self, ipts):
        out = np.zeros((ipts.shape[0],2))
        s = MinMaxScaler()
        ipts_norm = s.fit_transform(ipts)

        frame_tol = self.FRAME_MT_TOLERANCE / self.metaData['sample_freq']
        dist_tol = self.DIST_MT_TOLERANCE / (self.mean_size/2 +
                                             1 * self.metaData['meanDisplace'])

        for i in range(ipts.shape[0]):
            if ipts[i,1] <= 0 or ipts[i,0] > dist_tol or ipts[i,1] > frame_tol:
                score = 0
            else:
                score = np.round(1 - 0.25 * ipts_norm[i,0] - 0.25 * ipts_norm[i,1], 3)
            out[i,0] = 1-score
            out[i,1] = score
        return out

    def extract_train_from_break(self, sample_id, ipts, mt_dic):
        """Extract broken mitosis information to train model
        """
        sample = pd.DataFrame(sample_id)
        sample.columns = ['par', 'daug']
        idx = []
        for i in mt_dic.keys():
            par, daug = i, list(mt_dic[i]['daug'].keys())[0]
            sub = sample[(sample['par'] == par) & (sample['daug'] == daug)]
            if sub.shape[0] == 0:
                warnings.warn('Positive mitosis instance (parent-daughter) ' + str(par) + '-' + str(daug) + ' filtered out; your gating may be too strict.')
            else:
                idx.append(sub.index[0])

        idx = np.array(idx)
        return ipts[idx].copy()

    def associate(self, mode=None):
        ann = deepcopy(self.ann)
        track = deepcopy(self.track)
        mt_dic = deepcopy(self.mt_dic)

        parent_pool, pool = self.extract_pools()
        
        self.logger.debug('Short tracks excluded: ' + 
                          str(self.short_tracks)[1:-1])
        self.logger.debug('Candidate parents: ' + str(parent_pool)[1:-1])
        self.logger.debug('Candidate daughters: ' + str(pool)[1:-1])
        
        ipts, sample_id = self.extract_features(parent_pool, pool, remove_outlier=None, normalize=False)

        if mode is None or mode == 'SVM':

            # Train model further with already broken tracks
            ipts_brk = self.extract_train_from_break(sample_id, ipts, mt_dic)
            y = [1 for _ in range(ipts_brk.shape[0])]
            # Read in baseline training data
            baseline = np.array(pd.read_csv(self.SVM_PATH, header=None))
            # Merge baseline and broken data
            X = np.concatenate((ipts_brk, baseline[:, :baseline.shape[1] - 1]), axis=0)
            y.extend(list(baseline[:, baseline.shape[1] - 1]))
            y = np.array(y)

            # Oversample positive instances
            smote = BorderlineSMOTE(random_state=1, kind='borderline-1')
            X, y = smote.fit_resample(X, y)
            # Normalize
            s = RobustScaler()
            X = s.fit_transform(X)
            '''
            # Render training set to inspect
            save_train = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
            pd.DataFrame(save_train).to_csv('../test/test_train.csv', index=False, header=False)
            '''
            model = SVC(kernel='rbf', C=100, gamma=1, probability=True, class_weight='balanced')
            model.fit(X, y)

            s2 = RobustScaler()
            ipts_norm = s2.fit_transform(ipts)
            res = model.predict_proba(ipts_norm)
            '''
            # Render res and output prediction
            save_ipts = np.concatenate((ipts_norm, np.expand_dims(np.argmax(res, axis=1), axis=1)), axis=1)
            pd.DataFrame(save_ipts).to_csv('../test/test_res.csv', index=False, header=False)
            '''
        else:
            res = self.plainPredict(ipts)
        self.logger.info('Finished prediction.')

        parent_pool = list(np.unique(sample_id[:, 0]))
        cost_r_idx = np.array([val for val in parent_pool for _ in range(2)])
        cost_c_idx = np.unique(sample_id[:, 1])
        cost = np.zeros((cost_r_idx.shape[0], cost_c_idx.shape[0]))
        for i in range(cost.shape[0]):
            a = np.where(sample_id[:, 0] == cost_r_idx[i])[0].tolist()
            for j in range(cost.shape[1]):
                if cost_r_idx[i] != cost_c_idx[j]:
                    b = np.where(sample_id[:, 1] == cost_c_idx[j])[0].tolist()
                    sp_index = list(set(a) & set(b))
                    if sp_index:
                        if cost_r_idx[i] in self.mt_dic.keys() and \
                                cost_c_idx[j] in self.mt_dic[cost_r_idx[i]]['daug'].keys():
                            cost[i, j] = 1
                        else:
                            cost[i, j] = res[sp_index[0]][1]
        '''
        # Save input for debugging
        save_cost = pd.DataFrame(cost.copy())
        save_cost.index = cost_r_idx
        save_cost.columns = cost_c_idx
        save_cost.to_csv('../../test/test_cost.csv')
        pd.DataFrame(np.concatenate((ipts, sample_id, np.expand_dims(np.argmax(res, axis=1), axis=1)),
                                    axis=1)).to_csv('../../test/test_input.csv')
        '''
        cost = cost * -1
        row_ind, col_ind = linear_sum_assignment(cost)

        to_register = {}
        for i in range(len(row_ind)):
            cst = cost[row_ind[i], col_ind[i]]
            if cst < -0.5:
                par = cost_r_idx[row_ind[i]]
                daug = cost_c_idx[col_ind[i]]
                if par not in to_register.keys():
                    to_register[par] = ([daug], [cst])
                else:
                    to_register[par][0].append(daug)
                    to_register[par][1].append(cst)

        self.logger.debug('Parent-Daughter relation to register')
        self.logger.debug(to_register)
        #print(mt_dic)
        ips_count = 0
        for par in to_register.keys():
            daugs, csts = to_register[par]
            m_entry = self.mt_entry_lookup[par][0]
            if self.mt_entry_lookup[par][1] == 0:
                self.imprecise.append(par)
                ips_count += 1
            if par not in mt_dic.keys():
                for i in range(len(daugs)):
                    m_exit = self.mt_exit_lookup[daugs[i]][0]
                    if self.mt_exit_lookup[daugs[i]][1] == 0:
                        self.imprecise.append(daugs[i])
                        ips_count += 1
                    ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic),
                                                        par, daugs[i], m_exit, np.round(1+csts[i],3), m_entry)
            else:
                ori_daugs = list(mt_dic[par]['daug'].keys())
                for ori_daug in ori_daugs:
                    if ori_daug not in daugs:
                        ann, mt_dic = self.revert(deepcopy(ann), deepcopy(mt_dic), par, ori_daug)
                for i in range(len(daugs)):
                    if daugs[i] not in ori_daugs:
                        m_exit = self.mt_exit_lookup[daugs[i]][0]
                        if self.mt_exit_lookup[daugs[i]][1] == 0:
                            self.imprecise.append(daugs[i])
                            ips_count += 1
                        ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic),
                                                            par, daugs[i], m_exit, np.round(1+csts[i],3), m_entry)
                    else:
                        mt_dic[par]['daug'][daugs[i]]['dist'] = np.round(1+csts[i],3)

        # count 2 daughters-found relationships
        count = 0
        for i in mt_dic.keys():
            if len(list(mt_dic[i]['daug'].keys())) == 2:
                count += 1

        self.logger.info("Parent-Daughter-Daughter mitosis relations found: " + str(count))
        self.logger.info("Parent-Daughter mitosis relations found: " + str(len(list(mt_dic.keys())) - count))
        self.logger.info("Imprecise tracks involved in prediction: " + str(ips_count))
        #print(mt_dic)
        track = track.sort_values(by=['lineageId', 'trackId', 'frame'])
        return track, ann, mt_dic

    def update_table_with_mt(self):
        """Update tracked object table with information in self.mt_dic (mitosis lookup dict)
        """
        track = self.track.copy()
        dic = self.mt_dic.copy()
        for trk in list(dic.keys()):
            lin = track[track['trackId'] == trk]['lineageId'].iloc[0]
            for d in list(dic[trk]['daug'].keys()):
                track.loc[track['trackId'] == d, 'parentTrackId'] = trk
                track.loc[track['lineageId'] == d, 'lineageId'] = lin

        return track

    def smooth_track(self):
        """Re-assign cell cycle classification based on smoothed confidence
        """
        count = 0
        dic = {0: 'G1/G2', 1: 'S', 2: 'M'}
        track = self.track.copy()
        track_filtered = pd.DataFrame(columns=track.columns)
        flt = np.ones(self.SMOOTH)
        escape = int(np.floor(self.SMOOTH / 2))
        for i in np.unique(track['trackId']):
            cur_track = track[track['trackId'] == i].copy()
            if cur_track.shape[0] >= self.SMOOTH:
                S = np.convolve(cur_track['Probability of S'], flt, mode='valid') / self.SMOOTH
                M = np.convolve(cur_track['Probability of M'], flt, mode='valid') / self.SMOOTH
                G = np.convolve(cur_track['Probability of G1/G2'], flt, mode='valid') / self.SMOOTH
                ix = cur_track.index
                ix = ix[escape:(cur_track.shape[0] - escape)]
                cur_track.loc[ix, 'Probability of S'] = S
                cur_track.loc[ix, 'Probability of G1/G2'] = G
                cur_track.loc[ix, 'Probability of M'] = M
                idx = np.argmax(
                    np.array(cur_track.loc[:, ['Probability of G1/G2', 'Probability of S', 'Probability of M']]),
                    axis=1)
                phase = list(map(lambda x: dic[x], idx))
                count += np.sum(phase != cur_track['predicted_class'])
                cur_track.loc[:, 'predicted_class'] = phase

            track_filtered = track_filtered.append(cur_track.copy())
        self.logger.info("Object classification corrected by smoothing: " + str(count))

        return track

    def getMeanDisplace(self):
        """Calculate mean displace of each track normalized with frame

        Returns:
            (pandas.DataFrame): trackId, mean_displace
        """
        d = {'trackId': [], 'mean_displace': []}
        for i in np.unique(self.track['trackId']):
            sub = self.track[self.track['trackId'] == i]
            dp = []
            for j in range(1, sub.shape[0]):
                x1 = sub['Center_of_the_object_0'].iloc[j]
                y1 = sub['Center_of_the_object_1'].iloc[j]
                x2 = sub['Center_of_the_object_0'].iloc[j - 1]
                y2 = sub['Center_of_the_object_1'].iloc[j - 1]
                frame_diff = sub['frame'].iloc[j] - sub['frame'].iloc[j - 1]  # normalize with frame
                dp.append(dist(x1, y1, x2, y2) / frame_diff)
            if dp:
                d['mean_displace'].append(np.mean(dp))
                d['trackId'].append(i)

        return pd.DataFrame(d)

    def getMTscore(self, search_range, discount=0.9):
        """Measure mitosis score based on cell cycle classification

        Args:
            search_range (int): region for calculation
            discount (float): discounting factor when calculating mitosis score
                mitosis_score = SUM(discount^frame_diff * confidence of class 'M')
                default shallow discount = 0.9
        """
        PSEUDOCOUNT = 0.01

        mt_score_begin = {}
        mt_score_end = {}
        for trackId in np.unique(self.track['trackId']):
            sub = self.track[self.track['trackId'] == trackId]
            idx = sub.index[: np.min((sub.shape[0], search_range))]

            frame_start = sub['frame'].loc[idx[0]]
            x_start = np.power(discount, np.abs(sub['frame'].loc[idx] - frame_start))
            score_start = np.sum(
                np.multiply(x_start,
                            sub['Probability of M'].loc[idx] - sub['Probability of S'].loc[idx] +
                            int(sub['emerging'].loc[idx].values[0])))

            idx2 = sub.index[np.max((0, sub.shape[0] - search_range)):][::-1]
            frame_end = sub['frame'].loc[idx2[0]]
            x_end = np.power(discount, np.abs(sub['frame'].loc[idx2] - frame_end))
            score_end = np.sum(
                np.multiply(x_end,
                            sub['Probability of M'].loc[idx2] - sub['Probability of S'].loc[idx2]))

            mt_score_begin[trackId] = score_start / search_range + PSEUDOCOUNT
            mt_score_end[trackId] = score_end / search_range + PSEUDOCOUNT
        return mt_score_begin, mt_score_end

    def getSVMinput(self, parent, daughter):
        """Generate SVM classifier input for track 1 & 2

        Args:
            parent (int): parent track ID
            daughter (int): daughter track ID

        Returns:
            input vector of SVM classifier:
                [distance_diff, frame_diff, m_score_par + m_score_daug]    <-  track specific
                * ave_major/minor_axis_diff = abs(parent_axis - daughter_axis) / parent_axis
                some parameters are normalized with dataset specific features:
                    distance_diff /= ave_displace
                    frame_diff /= (sample_freq * mt_len)
        """
        par = self.track[self.track['trackId'] == parent].sort_values(by='frame')
        daug = self.track[self.track['trackId'] == daughter].sort_values(by='frame')

        # Feature 2: mitosis frame difference
        # For secondary mitosis, skip the frame before mitosis exit
        m_entry = self.getMtransition(parent, direction='entry')
        m_exit = self.getMtransition(daughter, direction='exit')
        if m_entry is None:
            m_entry = par['frame'].iloc[-1]
            x1 = par['Center_of_the_object_0'].iloc[-1]
            y1 = par['Center_of_the_object_1'].iloc[-1]
            self.mt_entry_lookup[parent] = (m_entry, 0)  # 0: imprecise
        else:
            idx = list(par['frame']).index(m_entry)
            x1 = par['Center_of_the_object_0'].iloc[idx]
            y1 = par['Center_of_the_object_1'].iloc[idx]
            self.mt_entry_lookup[parent] = (m_entry, 1)  # 1: precise
        if m_exit is None:
            m_exit = daug['frame'].iloc[0]
            x2 = daug['Center_of_the_object_0'].iloc[0]
            y2 = daug['Center_of_the_object_1'].iloc[0]
            self.mt_exit_lookup[daughter] = (m_exit, 0)
        else:
            idx = list(daug['frame']).index(m_exit)
            x2 = daug['Center_of_the_object_0'].iloc[idx]
            y2 = daug['Center_of_the_object_1'].iloc[idx]
            self.mt_exit_lookup[daughter] = (m_exit, 1)

        if m_entry >= daug['frame'].iloc[0]:
            # mitosis daughter should appear after NEBD of parent, set -1 to be filtered out in extract_feature() method
            frame_diff = -1
        else:
            frame_diff = m_exit - m_entry

        # Feature 1: distance
        distance_diff = dist(x1, y1, x2, y2)

        # Feature 3: mitosis score
        m_score_par = self.mt_score_end[parent]
        m_score_daug = self.mt_score_begin[daughter]

        # Feature 4: minimum intensity, pcna intensity during mitosi is very low
        par_intensity = par['mean_intensity'].iloc[-1]
        daug_intensity = daug['mean_intensity'].iloc[0]

        # Feature 5/6: length of parent and daughter track
        len_par = (par['frame'].iloc[-1] - par['frame'].iloc[0])/self.metaData['mt_len']
        len_daug = (daug['frame'].iloc[-1] - daug['frame'].iloc[0])/self.metaData['mt_len']

        out = [distance_diff / (self.mean_size/2 + np.abs(frame_diff) * self.metaData['meanDisplace']),
               frame_diff / self.metaData['sample_freq'],
               np.min((par_intensity, daug_intensity)) / self.mean_intensity,
               m_score_par + m_score_daug,
               np.min((len_par, len_daug)) / self.metaData['sample_freq']]

        return out

    def get_SVM_train(self, sample):
        """Save training data for SVM classifier of this particular dataset

        Args:
            sample (numpy.array): matrix of shape (sample, (parent ID, daughter ID, y))

        Returns:
            (numpy.array): Input feature map
            (numpy.array): Ground truth label
            (numpy.array): Track ID of the corresponding feature (row)
        """

        if not self.flag:
            raise NotImplementedError('Before extracting SVM training data, call doTrackRefine() first.')

        parent_pool, pool = self.extract_pools(extra_par=list(sample[:,0]), extra_daug=list(sample[:,1]))
        ipts, y, sample_id = self.extract_features(parent_pool, pool, sample=sample)

        return ipts, y, sample_id

    def setSVMpath(self, model_train):
        self.SVM_PATH = model_train
        return

    def doTrackRefine(self):
        """Perform track refinement process

        Returns:
            If run in TRH/SVM mode, will return annotation table, tracked object table and mitosis directory
            If run in TRAIN mode, will only return tracked object table after smoothing, mitosis breaking
                for manual inspection. After determining the training instance, generate training data through
                get_SVM_train(sample).
        """
        if self.flag:
            raise NotImplementedError('Do not call track refine object twice!')

        self.flag = True
        self.track = self.smooth_track()
        count = 0
        while True:
            count += 1
            self.logger.info('Level ' + str(count) + ' mitosis:')
            out = self.break_mitosis()
            self.track = out[0]
            if out[1] == 0:
                break

        self.track, self.short_tracks, self.ann = self.register_track()
        self.mt_score_begin, self.mt_score_end = self.getMTscore(self.SEARCH_RANGE, self.MT_DISCOUNT)
        if self.MODE == 'TRH':
            self.track, self.ann, self.mt_dic = self.associate(mode='TRH')
        elif self.MODE == 'SVM':
            if self.SVM_PATH == '':
                raise ValueError('Path to SVM training data has not set yet, use setSVMpath() to supply an SVM model.')
            self.track, self.ann, self.mt_dic = self.associate()
        elif self.MODE == 'TRAIN':
            return self.track, self.mt_dic

        self.track = self.update_table_with_mt()

        return self.ann, self.track, self.mt_dic, self.imprecise
