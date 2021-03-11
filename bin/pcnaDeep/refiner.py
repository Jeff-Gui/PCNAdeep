# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:03:45 2020

@author: Yifan Gui
"""

import pandas as pd
import numpy as np
import math
import re


def dist(x1, y1, x2, y2):
    return math.sqrt((float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2)


def deduce_transition(l, tar, confidence, min_tar, max_res, escape=0):
    """ Deduce mitosis exit and entry based on adaptive searching
        
        Args:
            l: list of the target cell cycle phase
            target: target cell cycle phase
            min_tar: minimum duration of an entire target phase
            confidence: matrix of confidence
            max_tar: maximum accumulative duration of unwanted phase
            escape: do not consider the first n instance
            
        Returns:
            (entry, exit), index of index list that resolved as entry and exit
        """
    mp = {'G1/G2': 0, 'S': 1, 'M': 2}
    confid_cls = list(map(lambda x: confidence[x, mp[l[x]]], range(confidence.shape[0])))
    idx = np.where(np.array(l) == tar)[0]
    idx = idx[idx >= escape].tolist()
    if len(idx) == 0: return None
    if len(idx) == 1: return idx[0], idx[0]
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
    elif i == (len(idx) - 1) and g_panelty < max_res and found == False and cur_m_entry != idx[-1]:
        found = True
        m_exit = idx[-1]

    if found:
        return cur_m_entry, m_exit
    else:
        return None


# The script does two things: find potential parent-daughter cells, and wipes out
# random false classification in the form of A-B-A
# To determine potential parent-daughter cells, appearance and disappearance time
# and location of the track are examined. Tracks appearing within certain 
# distance and time shift after another track's disappearance is considered as the daughter track.
# Parent-daughter track does not necessary mean mitosis event. In fact, it can
# be caused by either of the three events
# - 1. cells moving outside the view field and then come back, therefore assigned differently
# - 2. A mis-transition: Ilastik assign cells belong to the same lineage to two tracks
# - 3. Temporal loss of signal or segmentation issue
# - 4. Mitosis

class refiner:

    def __init__(self, track, threshold_F=60, threshold_mt_F=150, threshold_T=4, threshold_mt_T=5, smooth=5, minGS=3,
                 minM=3):
        self.track = track.copy()
        self.count = np.unique(track['trackId'])
        self.DIST_TOLERANCE = threshold_F
        self.FRAME_TOLERANCE = threshold_T
        self.FRAME_MT_TOLERANCE = threshold_mt_T
        self.DIST_MT_TOLERANCE = threshold_mt_F
        self.SMOOTH = smooth
        self.MIN_GS = minGS
        self.MIN_M = minM
        self.short_tracks = []
        self.mt_dic = {}
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
            sub = track[track['trackId'] == trk]
            if trk in list(self.mt_dic.keys()):
                filtered_track = filtered_track.append(sub.copy())
                continue

            found = False
            if sub.shape[0] > self.MIN_GS and 'M' in list(sub['predicted_class']):
                cls = sub['predicted_class'].tolist()
                confid = np.array(sub[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
                out = deduce_transition(l=cls, tar='M', confidence=confid, min_tar=self.MIN_M, max_res=self.MIN_GS,
                                        escape=self.MIN_M)

                if out is not None and out[0] != out[1] and out[1] != len(cls) - 1:
                    found = True
                    cur_m_entry, m_exit = out
                    # split mitosis track, keep parent track with 2 'M' prediction
                    # this makes cytokinesis unpredictable...
                    m_entry = list(sub['frame'])[cur_m_entry]
                    sp_time = list(sub['frame'])[m_exit]
                    new_track = sub[sub['frame'] >= sp_time].copy()
                    new_track.loc[:, 'trackId'] = cur_max
                    new_track.loc[:, 'lineageId'] = list(sub['lineageId'])[0]  # inherit the lineage
                    new_track.loc[:, 'parentTrackId'] = trk  # mitosis parent asigned
                    # register to the class
                    x1 = sub.iloc[m_exit]['Center_of_the_object_0']
                    y1 = sub.iloc[m_exit]['Center_of_the_object_1']
                    x2 = sub.iloc[m_exit - 1]['Center_of_the_object_0']
                    y2 = sub.iloc[m_exit - 1]['Center_of_the_object_1']
                    self.mt_dic[trk] = {'div': m_entry,
                                        'daug': {cur_max: {'m_exit': sp_time, 'dist': dist(x1, y1, x2, y2)}}}
                    cur_max += 1
                    count += 1
                    old_track = sub[sub['frame'] < sp_time].copy()
                    filtered_track = filtered_track.append(old_track.copy())
                    filtered_track = filtered_track.append(new_track.copy())
            if not found:
                filtered_track = filtered_track.append(sub.copy())

        print('Found mitosis track: ' + str(count))
        return filtered_track, count

    def register_track(self):
        """Register track annotation table 
        """

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
            if cur_track.shape[0] >= 2 * self.FRAME_MT_TOLERANCE:
                # record (dis-)appearance cell cycle classification, in time range equals to FRAME_TOLERANCE
                ann['app_stage'][i] = '-'.join(cur_track['predicted_class'].iloc[0:self.FRAME_MT_TOLERANCE])
                ann['disapp_stage'][i] = '-'.join(cur_track['predicted_class'].iloc[
                                                  (cur_track.shape[0] - self.FRAME_MT_TOLERANCE): cur_track.shape[0]])
            else:
                ann['app_stage'][i] = '-'.join(
                    cur_track['predicted_class'].iloc[0:min(self.FRAME_TOLERANCE, cur_track.shape[0])])
                ann['disapp_stage'][i] = '-'.join(
                    cur_track['predicted_class'].iloc[max(0, cur_track.shape[0] - self.FRAME_MT_TOLERANCE):])
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

    def compete(self, mt_dic, parentId, daughterId_1, dist1, daughterId_2, dist2):
        # check if a track ID can be registered into the mitosis
        # if a parent already have two parents, compete with distance
        """
        Args:
            mt_dic (dict): dictionary storing mitosis information
            parentId/daughterId_1/daughterId_2 (int): track IDs
            dist1/dist2:distance of parent and daughter tracks

        Returns:
            Dictionary: {register: id to register; revert: id to revert}
        """

        dg_list = mt_dic[parentId]['daug']
        ids = [daughterId_1, daughterId_2]
        dist = [dist1, dist2]
        for i in list(dg_list.keys()):
            if i not in ids:
                ids.append(i)
                dist.append(dg_list[i]['dist'])
        rs = np.argsort(dist)[:2]
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
        # remove information of a relationship registered to ann and mt_dic
        # parent
        mt_dic[parentId]['daug'].pop(daughterId)
        ori = ann.loc[ann['track'] == parentId]['mitosis_identity'].values[0]
        ori = ori[:re.search('/parent$', ori).span()[0]]
        ann.loc[ann['track'] == parentId, 'mitosis_identity'] = ori
        ori_daug = str(ann.loc[ann['track'] == parentId]['mitosis_daughter'].values[0])
        ori_daug = ori_daug.split('/')
        ori_daug.remove(str(daughterId))
        ann.loc[ann['track'] == parentId, 'mitosis_daughter'] = '/'.join(ori_daug)
        # daughter
        ori = ann.loc[ann['track'] == daughterId]['mitosis_identity'].values[0]
        ori = ori[:re.search('/daughter$', ori).span()[0]]
        ann.loc[ann['track'] == daughterId, 'mitosis_identity'] = ori
        ann.loc[ann['track'] == daughterId, 'm_exit'] = None
        ann.loc[ann['track'] == daughterId, 'mitosis_parent'] = None

        return ann, mt_dic

    def search_pdd(self):
        ann = self.ann.copy()
        track = self.track.copy()
        mt_dic = self.mt_dic.copy()

        count = 0
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
                                      ann['mitosis_identity'].iloc[x] == '', range(ann.shape[0])))]['track'])
                    else:
                        potential_parent = []
                        v1 = target_info_1['mitosis_parent'].values[0]
                        v2 = target_info_2['mitosis_parent'].values[0]
                        if v1 is not None: potential_parent.append(int(v1))
                        if v2 is not None: potential_parent.append(int(v2))
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
                                c1 = list(
                                    track[track['trackId'] == int(target_info_1['track'].values)]['predicted_class'])
                                c1_confid = np.array(track[track['trackId'] == int(target_info_1['track'].values)][
                                                         ['Probability of G1/G2', 'Probability of S',
                                                          'Probability of M']])
                                c2 = list(
                                    track[track['trackId'] == int(target_info_2['track'].values)]['predicted_class'])
                                c2_confid = np.array(track[track['trackId'] == int(target_info_2['track'].values)][
                                                         ['Probability of G1/G2', 'Probability of S',
                                                          'Probability of M']])
                                c1_exit = \
                                    deduce_transition(c1, tar='M', confidence=c1_confid, min_tar=1,
                                                      max_res=self.MIN_GS)[1]
                                c1_exit = list(track[track['trackId'] == int(target_info_1['track'].values)]['frame'])[
                                    c1_exit]
                                c2_exit = \
                                    deduce_transition(c2, tar='M', confidence=c2_confid, min_tar=1,
                                                      max_res=self.MIN_GS)[1]
                                c2_exit = list(track[track['trackId'] == int(target_info_2['track'].values)]['frame'])[
                                    c2_exit]

                                if ann.loc[ann['track'] == parent_id, "m_entry"].values[0] is None:
                                    # parent has not been registered yet
                                    c3 = list(track[track['trackId'] == parent_id]['predicted_class'])
                                    c3_class = c3
                                    c3_confid = np.array(track[track['trackId'] == parent_id][
                                                             ['Probability of G1/G2', 'Probability of S',
                                                              'Probability of M']])
                                    c3_entry = -(1 + deduce_transition(c3_class[::-1], tar='M',
                                                                       confidence=c3_confid[::-1, :], min_tar=1,
                                                                       max_res=self.MIN_GS)[1])
                                    c3_entry = list(track[track['trackId'] == parent_id]['frame'])[c3_entry]
                                    ann.loc[ann['track'] == parent_id, "m_entry"] = c3_entry
                                    # update information in ann table
                                    # daughter
                                    s1 = ann.loc[
                                        ann['track'] == potential_daughter_pair_id[i], "mitosis_identity"].values
                                    s2 = ann.loc[
                                        ann['track'] == potential_daughter_pair_id[j], "mitosis_identity"].values
                                    ann.loc[ann['track'] == potential_daughter_pair_id[
                                        i], "mitosis_identity"] = s1 + "/daughter"
                                    ann.loc[ann['track'] == potential_daughter_pair_id[
                                        j], "mitosis_identity"] = s2 + "/daughter"
                                    ann.loc[ann['track'] == potential_daughter_pair_id[i], "m_exit"] = c1_exit
                                    ann.loc[ann['track'] == potential_daughter_pair_id[j], "m_exit"] = c2_exit
                                    # parent
                                    ann.loc[ann['track'] == int(target_info_1['track']), "mitosis_parent"] = parent_id
                                    ann.loc[ann['track'] == int(target_info_2['track']), "mitosis_parent"] = parent_id
                                    s3 = ann.loc[ann['track'] == parent_id, "mitosis_identity"].values
                                    ann.loc[ann['track'] == parent_id, "mitosis_identity"] = s3 + "/parent" + "/parent"
                                    ann.loc[ann['track'] == parent_id, "mitosis_daughter"] = '/'.join(
                                        [str(target_info_1['track'].values[0]), str(target_info_2['track'].values[0])])

                                    mt_dic[parent_id] = {'div': c3_entry, 'daug': {}}
                                    mt_dic[parent_id]['daug'][potential_daughter_pair_id[i]] = {'m_exit': c1_exit,
                                                                                                'dist': dist_dif1}
                                    mt_dic[parent_id]['daug'][potential_daughter_pair_id[j]] = {'m_exit': c2_exit,
                                                                                                'dist': dist_dif2}

                                    count += 2
                                else:
                                    result = self.compete(mt_dic, parent_id, potential_daughter_pair_id[i], dist_dif1,
                                                          potential_daughter_pair_id[j], dist_dif2)

                                    for rv in result['revert']:
                                        ann, mt_dic = self.revert(ann, mt_dic, parent_id, rv)
                                        count -= 1
                                    for rg in result['register']:
                                        if rg == potential_daughter_pair_id[i]:
                                            ori = ann.loc[ann['track'] == parent_id, "mitosis_daughter"].values[0]
                                            ori_idt = ann.loc[ann['track'] == parent_id, "mitosis_identity"].values[0]
                                            s1 = ann.loc[ann['track'] == potential_daughter_pair_id[
                                                i], "mitosis_identity"].values
                                            ann.loc[ann['track'] == potential_daughter_pair_id[
                                                i], "mitosis_identity"] = s1 + "/daughter"
                                            ann.loc[ann['track'] == potential_daughter_pair_id[i], "m_exit"] = c1_exit
                                            ann.loc[ann['track'] == int(
                                                target_info_1['track']), "mitosis_parent"] = parent_id
                                            ann.loc[ann['track'] == parent_id, "mitosis_daughter"] = str(
                                                ori) + '/' + str(target_info_1['track'].values[0])
                                            ann.loc[ann['track'] == parent_id, "mitosis_identity"] = str(
                                                ori_idt) + '/parent'
                                            mt_dic[parent_id]['daug'][int(target_info_1['track'])] = {'m_exit': c1_exit,
                                                                                                      'dist': dist_dif1}
                                        elif rg == potential_daughter_pair_id[j]:
                                            ori = ann.loc[ann['track'] == parent_id, "mitosis_daughter"].values[0]
                                            ori_idt = ann.loc[ann['track'] == parent_id, "mitosis_identity"].values[0]
                                            s2 = ann.loc[ann['track'] == potential_daughter_pair_id[
                                                j], "mitosis_identity"].values
                                            ann.loc[ann['track'] == potential_daughter_pair_id[
                                                j], "mitosis_identity"] = s2 + "/daughter"
                                            ann.loc[ann['track'] == potential_daughter_pair_id[j], "m_exit"] = c2_exit
                                            ann.loc[ann['track'] == int(
                                                target_info_2['track']), "mitosis_parent"] = parent_id
                                            ann.loc[ann['track'] == parent_id, "mitosis_daughter"] = str(
                                                ori) + '/' + str(target_info_2['track'].values[0])
                                            ann.loc[ann['track'] == parent_id, "mitosis_identity"] = str(
                                                ori_idt) + '/parent'
                                            mt_dic[parent_id]['daug'][int(target_info_2['track'])] = {'m_exit': c2_exit,
                                                                                                      'dist': dist_dif2}

                                        count += 1

        print("Parent-Daughter-Daughter mitosis relations found: " + str(count))
        track = track.sort_values(by=['lineageId', 'trackId', 'frame'])
        return track, ann, mt_dic

    def update_table_with_mt(self):
        track = self.track.copy()
        dic = self.mt_dic.copy()
        for trk in list(dic.keys()):
            lin = track[track['trackId'] == trk]['lineageId'].iloc[0]
            for d in list(dic[trk]['daug'].keys()):
                track.loc[track['trackId'] == d, 'parentTrackId'] = trk
                track.loc[track['lineageId'] == d, 'lineageId'] = lin

        return track

    def smooth_track(self):
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

    def doTrackRefine(self):

        self.track = self.smooth_track()
        while True:
            out = self.break_mitosis()
            self.track = out[0]
            if out[1] == 0:
                break

        self.track, self.short_tracks, self.ann = self.register_track()
        self.track, self.ann, self.mt_dic = self.search_pdd()
        self.track = self.update_table_with_mt()

        return self.ann, self.track, self.mt_dic
