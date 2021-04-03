# -*- coding: utf-8 -*-
import math
import re
import joblib
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from pcnaDeep.data.utils import get_outlier


def dist(x1, y1, x2, y2):
    """Calculate distance of a set of coordinates
    """
    return math.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2)


def deduce_transition(l, tar, confidence, min_tar, max_res, escape=0):
    """ Deduce mitosis exit and entry based on adaptive searching
        
        Args:
            l (list): list of the target cell cycle phase
            tar (str): target cell cycle phase
            min_tar (int): minimum duration of an entire target phase
            confidence (numpy.array): matrix of confidence
            max_res (int): maximum accumulative duration of unwanted phase
            escape (int): do not consider the first n instances
            
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
    elif i == (len(idx) - 1) and g_panelty < max_res and not found and cur_m_entry != idx[-1]:
        found = True
        m_exit = idx[-1]
        if m_exit - cur_m_entry + 1 < min_tar:
            return None

    if found:
        return cur_m_entry, m_exit
    else:
        return None


class Refiner:

    def __init__(self, track, smooth=5, minGS=6, minM=3, mode='SVM',
                 threshold_mt_F=150, threshold_mt_T=5,
                 search_range=5, mt_len=5, sample_freq=20, model_path=''):
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
                model_path (str): path to saved SVM model

        """
        self.flag = False
        self.track = track.copy()
        self.count = np.unique(track['trackId'])

        self.MODE = mode
        if mode == 'SVM' or mode == 'TRAIN':
            self.SEARCH_RANGE = search_range
            self.MT_DISCOUNT = 0.9
            self.metaData = {'mt_len': mt_len, 'sample_freq': sample_freq,
                             'meanDisplace': np.mean(self.getMeanDisplace()['mean_displace'])}
            self.mt_score_begin, self.mt_score_end = self.getMTscore(self.SEARCH_RANGE, self.MT_DISCOUNT)
            self.SVM_PATH = model_path
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
        self.mt_dic = {}
        self.mean_axis_lookup = {}
        self.mean_size = np.mean(np.array(self.track[['major_axis', 'minor_axis']]))
        self.imprecise = []  # imprecise mitosis: daughter exit without M classification
        # mitosis dictionary: trackId: [[daughterId], [mt_entry, mt_exit]] for primarily mitosis break only  {
        # 'div':m_entry, 'daug':{daug1: m_exit, daug2: m_exit}}
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
                                        'daug': {cur_max: {'m_exit': frame_list[m_exit], 'dist': dist(x1, y1, x2, y2)}}}
                    cur_max += 1
                    count += 1
                    old_track = sub[sub['frame'] < frame_list[sp_time]].copy()
                    filtered_track = filtered_track.append(old_track.copy())
                    filtered_track = filtered_track.append(new_track.copy())
            if not found:
                filtered_track = filtered_track.append(sub.copy())

        print('Found mitosis track: ' + str(count))
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

            if cur_track.shape[0] < 2 * frame_tolerance:
                short_tracks.append(i)

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
        print("High quality tracks subjected to predict relationship: " + str(ann.shape[0] - len(short_tracks)))

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

    def compete(self, mt_dic, parentId, daughter_ids, distance):
        # check if a track ID can be registered into the mitosis
        # if a parent already have two parents, compete with distance
        """
        Args:
            mt_dic (dict): dictionary storing mitosis information
            parentId (int): parent track ID
            daughter_ids (list): daughter_ids to compete
            distance (list): daughter_ids and corresponding distance / score-to-minimize

        Returns:
            Dictionary: {register: id to register; revert: id to revert}
        """

        dg_list = mt_dic[parentId]['daug']
        ids = daughter_ids
        for i in list(dg_list.keys()):
            if i not in ids:
                ids.append(i)
                distance.append(dg_list[i]['dist'])
        rs = np.argsort(distance)[:2]
        ids = [ids[rs[0]], ids[rs[1]]]
        rt = ids.copy()

        if ids[0] in dg_list.keys():
            rt.remove(ids[0])
        if ids[1] in dg_list.keys():
            rt.remove(ids[1])

        rm = []
        for i in list(dg_list.keys()):
            if i not in ids:
                rm.append(i)

        return {'register': rt, 'revert': rm}

    def revert(self, ann, mt_dic, parentId, daughterId):
        """Remove information of a relationship registered to ann and mt_dic
        """
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

    def getMtransition(self, trackId, direction='entry'):
        """Get mitosis transition time by trackId

        Args:
            trackId (int): track ID
            direction (str): either 'entry' or 'exit', mitosis entry or exit
        """
        c1 = list(self.track[self.track['trackId'] == trackId]['predicted_class'])
        c1_confid = np.array(self.track[self.track['trackId'] == trackId][
                                 ['Probability of G1/G2', 'Probability of S', 'Probability of M']])
        if direction == 'exit':
            trans = deduce_transition(c1, tar='M', confidence=c1_confid, min_tar=1, max_res=self.MIN_GS)
            if trans is not None:
                trans = trans[1]
            else:
                return None
        elif direction == 'entry':
            trans = deduce_transition(c1[::-1], tar='M', confidence=c1_confid[::-1, :],
                                      min_tar=1, max_res=self.MIN_GS)
            if trans is not None:
                trans = -(1 + trans[1])
            else:
                return None
        else:
            raise ValueError('Direction can either be entry or exit')

        trans = list(self.track[self.track['trackId'] == trackId]['frame'])[trans]

        return trans

    def svm_pdd(self):
        ann = deepcopy(self.ann)
        track = deepcopy(self.track)
        mt_dic = deepcopy(self.mt_dic)
        # deduce candidate parents = 
        #   mt_dic + wild parents not being parent.daughter yet
        parent_pool = list(self.mt_dic.keys())
        pool = list(np.unique(self.track['trackId']))
        lin_par_pool = np.unique(self.track['parentTrackId'])
        lin_daug_pool = np.unique(self.track[self.track['parentTrackId'] > 0]['trackId'])
        for i in pool:
            if i not in parent_pool and i not in lin_par_pool and i not in lin_daug_pool and i not in self.short_tracks:
                # wild parents: at least two M classification at the end
                if re.search('M', ann[ann['track'] == i]['disapp_stage'].values[0]) is not None :
                    parent_pool.append(i)

        print('Extracting features...')
        ft = 0
        ipts = []
        sample_id = []
        for i in parent_pool:
            for j in range(len(pool)):
                if i != pool[j]:
                    if ft % 1000 == 0 and ft > 0:
                        print('Considered ' + str(ft) + '/' + str(len(pool) * len(parent_pool)) + ' cases.')
                    ft += 1
                    ipts.append(self.getSVMinput(i, pool[j]))
                    sample_id.append([i, pool[j]])
        ipts = np.array(ipts)
        sample_id = np.array(sample_id)

        cls = joblib.load(self.SVM_PATH)
        # remove outlier
        outs = get_outlier(ipts, col_ids=[0])
        idx = [_ for _ in range(ipts.shape[0]) if _ not in outs]
        ipts = ipts[idx,]
        sample_id = sample_id[idx,]
        print('Removed outliers, remaining: ' + str(ipts.shape[0]))
        # normalization
        scaler = StandardScaler()
        scaler.fit(ipts)
        ipts = scaler.transform(ipts)
        
        res = cls.predict_proba(ipts)
        print('Finished prediction.')
        
        pred_pos = list(np.argmax(res, axis=1))
        for i in range(len(pred_pos)):
            if sample_id[i,0] in mt_dic.keys():
                if sample_id[i,1] in mt_dic[sample_id[i,0]]['daug'].keys():
                    pred_pos[i] += 0.5
        import matplotlib.pyplot as plt
        plt.scatter(ipts[:,0], ipts[:,1], c=pred_pos, s=(- min(ipts[:,2]) + ipts[:,2])*5, alpha=0.2, cmap='brg')
        plt.savefig('./test.jpg')
        #print(mt_dic)
        #print(sample_id[np.where(np.argmax(res, axis=1)==1)[0]])

        parent_pool = list(np.unique(sample_id[:, 0]))
        cost_r_idx = np.array([val for val in parent_pool for i in range(2)])
        cost_c_idx = np.unique(sample_id[:, 1])
        cost = np.zeros((cost_r_idx.shape[0], cost_c_idx.shape[0]))
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                if cost_r_idx[i] != cost_c_idx[j]:
                    a = np.where(sample_id[:, 0] == cost_r_idx[i])[0].tolist()
                    b = np.where(sample_id[:, 1] == cost_c_idx[j])[0].tolist()
                    sp_index = list(set(a) & set(b))
                    if sp_index:
                        cost[i, j] = res[sp_index[0]][1]

        cost = cost * -1
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_par = cost_r_idx[row_ind[::2]]
        anns = []
        for i in range(len(matched_par)):
            anns.append(ann)
            cst = cost[2 * i + 1, col_ind[2 * i:2 * i + 2]]
            par = matched_par[i]
            if all(cst < -0.5):
                daugs = cost_c_idx[col_ind[2 * i:2 * i + 2]]
                cst = (1 + cst).tolist()
                if par in list(mt_dic.keys()):
                    ori_daugs = list(mt_dic[par]['daug'].keys())
                    for j in range(len(ori_daugs)):
                        if ori_daugs[j] not in daugs:
                            ann, mt_dic = self.revert(deepcopy(ann), deepcopy(mt_dic), par, ori_daugs[j])
                    for j in range(len(daugs)):
                        daug = daugs[j]
                        if daug in mt_dic[par]['daug'].keys():  # update daughter distance as daughter confidence
                            mt_dic[par]['daug'][daug]['dist'] = cst[j]
                        else:
                            m_exit = self.getMtransition(daug, direction='exit')
                            if m_exit is None:
                                m_exit = self.track[self.track['trackId'] == daug]['frame'].iloc[0]
                                if m_exit <= mt_dic[par]['div']:
                                    continue
                                self.imprecise.append(daug)
                            if m_exit <= mt_dic[par]['div']:
                                continue
                            ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic), par, daug, m_exit,
                                                                cst[j])

                else:
                    m_entry = self.getMtransition(par, direction='entry')
                    for j in range(len(daugs)):
                        m_exit = self.getMtransition(daugs[j], direction='exit')
                        if m_exit is None:
                            m_exit = self.track[self.track['trackId'] == daugs[j]]['frame'].iloc[0]
                            if m_exit <= m_entry:
                                continue
                            self.imprecise.append(daugs[j])
                        if m_exit <= m_entry:
                            continue
                        ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic), par, daugs[j], m_exit,
                                                            cst[j], m_entry)
            else:
                if par in list(mt_dic.keys()):
                    daugs = list(mt_dic[par]['daug'].keys())
                    for daug in daugs:
                        ann, mt_dic = self.revert(deepcopy(ann), deepcopy(mt_dic), par, daug)
                    del mt_dic[par]

        # calculate 2 daughters found relationships
        count = 0
        for i in mt_dic.keys():
            if len(list(mt_dic[i]['daug'].keys())) == 2:
                count += 1

        print("Parent-Daughter-Daughter mitosis relations found: " + str(count))
        print("Parent-Daughter mitosis relations found: " + str(len(list(mt_dic.keys())) - count))
        track = track.sort_values(by=['lineageId', 'trackId', 'frame'])
        return track, ann, mt_dic

    def search_pdd(self):
        ann = deepcopy(self.ann)
        track = deepcopy(self.track)
        mt_dic = deepcopy(self.mt_dic)

        # Mitosis search
        #   Aim: to identify two appearing daughter tracks after one disappearing parent track
        #   Algorithm: find potential daughters, for each pair of them, 
        potential_daughter_pair_id = list(
            ann[list(map(lambda x: re.search('M', ann['app_stage'].iloc[x]) is not None, range(ann.shape[0])))][
                'track'])  # daughter track must appear as M during mitosis
        for i in range(len(potential_daughter_pair_id) - 1):
            for j in range(i + 1, len(potential_daughter_pair_id)):
                # iterate over all pairs of potential daughters
                target_info_1 = ann[ann['track'] == potential_daughter_pair_id[i]]
                target_info_2 = ann[ann['track'] == potential_daughter_pair_id[j]]
                if target_info_1.shape[0] == 0 or target_info_2.shape[1] == 0: continue
                time_dif = abs(int(target_info_1['app_frame']) - int(target_info_2['app_frame']))
                if dist(target_info_1['app_x'], target_info_1['app_y'], target_info_2['app_x'], target_info_2[
                        'app_y']) <= time_dif * self.DIST_MT_TOLERANCE and time_dif < self.FRAME_MT_TOLERANCE:
                    # Constraint A: close distance
                    # Constraint B: close appearing time

                    # Find potential parent that disappear at M
                    if target_info_1['mitosis_parent'].values[0] is None and target_info_2['mitosis_parent'].values[
                            0] is None:
                        potential_parent = list(ann[list(map(
                            lambda x: re.search('M', ann['disapp_stage'].iloc[x]) is not None and
                            ann['mitosis_identity'].iloc[x] == '' and x not in self.short_tracks,
                            range(ann.shape[0])))]['track'])
                    else:
                        potential_parent = []
                        v1 = target_info_1['mitosis_parent'].values[0]
                        v2 = target_info_2['mitosis_parent'].values[0]
                        if v1 is not None:
                            potential_parent.append(int(v1))
                        if v2 is not None:
                            potential_parent.append(int(v2))
                        if (v1 is not None) and (v2 is not None): continue

                    for k in range(len(potential_parent)):
                        if potential_parent[k] == potential_daughter_pair_id[i] or potential_parent[k] == \
                                potential_daughter_pair_id[j]:
                            continue

                        # spatial condition
                        parent_x = int(ann[ann['track'] == potential_parent[k]]["disapp_x"])
                        parent_y = int(ann[ann['track'] == potential_parent[k]]["disapp_y"])
                        parent_disapp_time = int(ann[ann['track'] == potential_parent[k]]["disapp_frame"])
                        parent_id = int(ann[ann['track'] == potential_parent[k]]["track"])

                        time_dif1 = abs(int(target_info_1['app_frame']) - parent_disapp_time)
                        time_dif2 = abs(int(target_info_2['app_frame']) - parent_disapp_time)
                        dist_dif1 = dist(target_info_1['app_x'], target_info_1['app_y'], parent_x, parent_y)
                        dist_dif2 = dist(target_info_1['app_x'], target_info_2['app_y'], parent_x, parent_y)

                        if dist_dif1 <= self.DIST_MT_TOLERANCE and dist_dif2 <= self.DIST_MT_TOLERANCE:
                            # Note, only one distance constaint met is accepted.
                            # Constraint A: parent close to both daughter tracks' appearance
                            if time_dif1 < self.FRAME_MT_TOLERANCE and time_dif2 < self.FRAME_MT_TOLERANCE:
                                # Constraint B: parent disappearance time close to daughter's appearance
                                # deduce M_entry and M_exit
                                c1_exit = self.getMtransition(potential_daughter_pair_id[i], direction='exit')
                                c2_exit = self.getMtransition(potential_daughter_pair_id[j], direction='exit')

                                if ann.loc[ann['track'] == parent_id, "m_entry"].values[0] is None:
                                    # parent has not been registered yet
                                    c3_entry = self.getMtransition(parent_id, direction='entry')

                                    # update information in ann and mt_dic table
                                    ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic), parent_id,
                                                                        potential_daughter_pair_id[i],
                                                                        c1_exit, dist_dif1, m_entry=c3_entry)
                                    ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic), parent_id,
                                                                        potential_daughter_pair_id[j],
                                                                        c2_exit, dist_dif2)

                                else:
                                    result = self.compete(mt_dic, parent_id, [potential_daughter_pair_id[i],
                                                                              potential_daughter_pair_id[j]],
                                                          [dist_dif1, dist_dif2])

                                    for rv in result['revert']:
                                        ann, mt_dic = self.revert(deepcopy(ann), deepcopy(mt_dic), parent_id, rv)
                                    for rg in result['register']:
                                        if rg == potential_daughter_pair_id[i]:
                                            ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic),
                                                                                parent_id, rg, c1_exit, dist_dif1)
                                        elif rg == potential_daughter_pair_id[j]:
                                            ann, mt_dic = self.register_mitosis(deepcopy(ann), deepcopy(mt_dic),
                                                                                parent_id, rg, c2_exit, dist_dif2)

        # calculate 2 daughters found relationships
        count = 0
        for i in mt_dic.keys():
            if len(list(mt_dic[i]['daug'].keys())) == 2:
                count += 1

        print("Parent-Daughter-Daughter mitosis relations found: " + str(count))
        print("Parent-Daughter mitosis relations found: " + str(len(list(mt_dic.keys())) - count))
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
        print("Object classification corrected by smoothing: " + str(count))

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
        mt_score_begin = {}
        mt_score_end = {}
        for trackId in np.unique(self.track['trackId']):
            sub = self.track[self.track['trackId'] == trackId]
            idx = sub.index[: np.min((sub.shape[0], search_range))]

            frame_start = sub['frame'].loc[idx[0]]
            x_start = np.power(discount, np.abs(sub['frame'].loc[idx] - frame_start))
            score_start = np.sum(
                np.multiply(x_start, sub['Probability of M'].loc[idx] - sub['Probability of S'].loc[idx]))

            idx2 = sub.index[np.max((0, sub.shape[0] - search_range)):][::-1]
            frame_end = sub['frame'].loc[idx2[0]]
            x_end = np.power(discount, np.abs(sub['frame'].loc[idx2] - frame_end))
            score_end = np.sum(
                np.multiply(x_end, sub['Probability of M'].loc[idx2] - sub['Probability of S'].loc[idx2]))

            mt_score_begin[trackId] = score_start / search_range + 0.1
            mt_score_end[trackId] = score_end / search_range + 0.1
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

        x1 = par['Center_of_the_object_0'].iloc[-1]
        y1 = par['Center_of_the_object_1'].iloc[-1]
        x2 = daug['Center_of_the_object_0'].iloc[0]
        y2 = daug['Center_of_the_object_1'].iloc[0]
        distance_diff = dist(x1, y1, x2, y2)
        frame_diff = np.abs(par['frame'].iloc[-1] - daug['frame'].iloc[0])
        m_score_par = self.mt_score_end[parent]
        m_score_daug = self.mt_score_begin[daughter]

        out = [distance_diff / (self.mean_size + np.abs(frame_diff) * self.metaData['meanDisplace']),
               1 / (frame_diff / (self.metaData['sample_freq'] * self.metaData['mt_len']) + 0.1),
               m_score_par + m_score_daug]

        return out

    def get_SVM_train(self, sample):
        """Save training data for SVM classifier of this particular dataset

        Args:
            sample (numpy.array): matrix of shape (sample, (parent ID, daughter ID, y))

        Returns:
            (numpy.array): X, y
        """
        self.track, self.short_tracks, self.ann = self.register_track()

        X = []
        y = []
        dic = {}
        for i in range(sample.shape[0]):
            r = sample[i, :]
            X.append(self.getSVMinput(r[0], r[1]))
            if r[0] in list(dic.keys()):
                dic[r[0]].append(r[1])
            else:
                dic[r[0]] = [r[1]]
            y.append(r[2])

        ann = deepcopy(self.ann)
        # deduce candidate parents = 
        #   mt_dic + wild parents not being parent.daughter yet
        parent_pool = list(np.unique((sample[:,0])))
        pool = list(np.unique(self.track['trackId']))
        lin_par_pool = np.unique(self.track['parentTrackId'])
        lin_daug_pool = np.unique(self.track[self.track['parentTrackId'] > 0]['trackId'])
        for i in pool:
            if i not in parent_pool and i not in lin_par_pool and i not in lin_daug_pool and i not in self.short_tracks:
                # wild parents: at least two M classification at the end
                if re.search('M', ann[ann['track'] == i]['disapp_stage'].values[0]) is not None:
                    parent_pool.append(i)

        print('Extracting features...')
        ft = 0
        ipts = []
        y = []
        for i in parent_pool:
            for j in range(len(pool)):
                if i != pool[j]:
                    if ft % 1000 == 0 and ft > 0:
                        print('Considered ' + str(ft) + '/' + str(len(pool) * len(parent_pool)) + ' cases.')
                    ft += 1
                    ipts.append(self.getSVMinput(i, pool[j]))
                    a = np.where(sample[:, 0] == i)[0].tolist()
                    b = np.where(sample[:, 1] == pool[j])[0].tolist()
                    sp_index = list(set(a) & set(b))
                    if sp_index:
                        y.append(1)
                    else:
                        y.append(0)

        ipts = np.array(ipts)
        y = np.array(y)

        return ipts, y

    def setSVMpath(self, model_path):
        self.SVM_PATH = model_path
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
        count = 1
        while True:
            print('Level ' + str(count) + ' mitosis:')
            count += 1
            out = self.break_mitosis()
            self.track = out[0]
            if out[1] == 0:
                break

        self.track, self.short_tracks, self.ann = self.register_track()
        if self.MODE == 'TRH':
            self.track, self.ann, self.mt_dic = self.search_pdd()
        elif self.MODE == 'SVM':
            self.mt_score_begin, self.mt_score_end = self.getMTscore(self.SEARCH_RANGE, self.MT_DISCOUNT)
            if self.SVM_PATH == '':
                raise ValueError('SVM model path has not set yet, use setSVMpath() to supply a SVM model.')
            self.track, self.ann, self.mt_dic = self.svm_pdd()
        elif self.MODE == 'TRAIN':
            self.mt_score_begin, self.mt_score_end = self.getMTscore(self.SEARCH_RANGE, self.MT_DISCOUNT)
            return self.track, self.mt_dic

        self.track = self.update_table_with_mt()

        return self.ann, self.track, self.mt_dic, self.imprecise
