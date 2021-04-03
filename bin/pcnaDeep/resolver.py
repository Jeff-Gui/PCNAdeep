# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pcnaDeep.refiner import deduce_transition


def list_dist(a, b):
    """Count different between elements of two lists
    a (list): original cls
    b (list): resolved cls
    """
    count = 0
    assert len(a) == len(b)
    for i in range(len(a)):
        if a[i] != b[i]:
            count += 1
        if a[i] == 'G1/G2' and (b[i] == 'G1' or b[i] == 'G2'):
            count -= 1

    return count


class Resolver:

    def __init__(self, track, ann, mt_dic, minG=6, minS=5, minM=3, minTrack=10, impreciseExit=None):
        if impreciseExit is None:
            impreciseExit = []
        self.impreciseExit = impreciseExit
        self.track = track
        self.ann = ann
        self.minG = minG
        self.minS = minS
        self.mt_dic = mt_dic
        self.minM = minM
        self.rsTrack = None
        self.minTrack = minTrack
        self.unresolved = []
        self.phase = pd.DataFrame(columns=['track', 'type', 'G1', 'S', 'M', 'G2', 'parent'])

    def doResolve(self):
        """Resolve cell cycle duration, identify G1 or G2
        
        Main function of class resolver
        
        Returns:
            1. track table with additional column 'resolved_class'
            2. phase table with cell cycle durations
        """

        track = self.track.copy()
        rt = pd.DataFrame()
        for i in np.unique(track['lineageId']):
            d = track[track['lineageId'] == i]
            t = self.resolveLineage(d, i)
            rt = rt.append(t)

        rt = rt.sort_values(by=['trackId', 'frame'])
        self.rsTrack = rt.copy()
        phase = self.doResolvePhase()
        return rt, phase

    def resolveLineage(self, lineage, main):
        """Resolve all tracks in a lineage recursively
        main (int): the parent track ID of current search
        """

        info = self.ann.loc[self.ann['track'] == main]
        m_entry = info['m_entry'].values[0]
        m_exit = info['m_exit'].values[0]

        if len(np.unique(lineage['trackId'])) == 1:
            return self.resolveTrack(lineage.copy(), m_entry=m_entry, m_exit=m_exit)
        else:
            out = pd.DataFrame()
            lg = lineage[lineage['trackId'] == main]
            out = out.append(self.resolveTrack(lg.copy(), m_entry=m_entry, m_exit=m_exit))
            daugs = self.mt_dic[main]['daug']
            for i in list(daugs.keys()):
                out = out.append(
                    self.resolveLineage(lineage[(lineage['trackId'] == i) | (lineage['parentTrackId'] == i)].copy(), i))
            return out

    def resolveTrack(self, trk, m_entry=None, m_exit=None):
        """Resolve single track.
        
        Args:
            trk (pandas.DataFrame): track table
            m_entry (int): time of mitosis entry corresponding to 'frame' column in table
            m_exit (int): time of mitosis exit corresponding to 'frame' column in table
            
                If no m time supplied, only treat as G1/G2/S track.
                Arrested track not resolved, return full G1/G2 list.
            
        Returns:
            pandas.DataFrame table with addition column of resolved class
        """

        UNRESOLVED_FRACTION = 0.2  # after resolving the class, if more than x% class has been corrected, label with
        # unresolved
        resolved_class = ['G1/G2' for _ in range(trk.shape[0])]

        cls = trk['predicted_class'].tolist()
        confid = np.array(trk[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
        out = deduce_transition(l=cls, tar='S', confidence=confid, min_tar=self.minS,
                                max_res=np.max((self.minM, self.minG)))

        if not (out is None or out[0] == out[1]):
            a = (out[0], np.min((out[1] + 1, len(resolved_class) - 1)))
            resolved_class[a[0]:a[1] + 1] = ['S' for _ in range(a[0], a[1] + 1)]

            if a[0] > 0:
                resolved_class[:a[0]] = ['G1' for _ in range(a[0])]
            if a[1] < len(resolved_class) - 1:
                resolved_class[a[1]:] = ['G2' for _ in range(len(resolved_class) - a[1])]

        frame = trk['frame'].tolist()
        if m_exit is not None:
            resolved_class[:frame.index(m_exit) + 1] = ['M' for _ in range(frame.index(m_exit) + 1)]
            i = frame.index(m_exit) + 1
            while i < len(resolved_class):
                if resolved_class[i] == 'G1/G2':
                    resolved_class[i] = 'G1'
                else:
                    break
                i += 1
        if m_entry is not None:
            resolved_class[frame.index(m_entry):] = ['M' for _ in range(len(resolved_class) - frame.index(m_entry))]
            i = frame.index(m_entry) - 1
            while i >= 0:
                if resolved_class[i] == 'G1/G2':
                    resolved_class[i] = 'G2'
                else:
                    break
                i -= 1

        if m_exit is None and m_entry is None:
            # some tracks begin/end with mitosis and not associated during refinement. In this case, override any
            # classification at terminal
            # WARNING: this can leads to false negative
            mt_out_begin = deduce_transition(l=cls, tar='M', confidence=confid, min_tar=1,
                                             max_res=np.max((self.minS, self.minG)))
            mt_out_end = deduce_transition(l=cls[::-1], tar='M', confidence=confid[::-1, :], min_tar=1,
                                           max_res=np.max((self.minS, self.minG)))

            if mt_out_begin is not None and mt_out_end is None:
                if mt_out_begin[0] == 0:
                    resolved_class[mt_out_begin[0]: mt_out_begin[1] + 1] = ['M' for _ in
                                                                            range(mt_out_begin[0], mt_out_begin[1] + 1)]
                # if followed with G1/G2 only, change to G1
                if np.unique(resolved_class[mt_out_begin[1] + 1:]).tolist() == ['G1/G2']:
                    resolved_class = ['G1' if i == 'G1/G2' else i for i in resolved_class]
            if mt_out_end is not None and mt_out_begin is None:
                if mt_out_end[0] == 0:
                    resolved_class = resolved_class[::-1]
                    resolved_class[mt_out_end[0]: mt_out_end[1] + 1] = ['M' for _ in
                                                                        range(mt_out_end[0], mt_out_end[1] + 1)]
                    if np.unique(resolved_class[mt_out_end[1] + 1:]).tolist() == ['G1/G2']:
                        resolved_class = ['G2' if i == 'G1/G2' else i for i in resolved_class]
                    resolved_class = resolved_class[::-1]

        trk['resolved_class'] = resolved_class
        if list_dist(cls, resolved_class) > UNRESOLVED_FRACTION * len(resolved_class):
            if len(resolved_class) >= self.minTrack:
                print('Too different, check: ' + str(trk['trackId'].tolist()[0]))
            self.unresolved.append(trk['trackId'].tolist()[0])
        return trk

    def doResolvePhase(self):
        """Resolve phase durations
        """
        out = {'track': [], 'type': [], 'length': [], 'arrest': [], 'G1': [], 'S': [], 'M': [], 'G2': [], 'parent': []}

        # register tracks
        for i in range(self.ann.shape[0]):
            info = self.ann.loc[i, :]
            if info['track'] in self.unresolved: continue
            sub = self.rsTrack[self.rsTrack['trackId'] == info['track']]
            length = np.max(sub['frame']) - np.min(sub['frame'])
            par = info['mitosis_parent']
            if par is None: par = 0
            out['track'].append(info['track'])
            out['length'].append(length)
            out['parent'].append(par)
            out['M'].append(np.nan)  # resolve later

            if np.unique(sub['resolved_class']).tolist() == ['G1/G2']:
                out['type'].append('arrest' + '-G')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif np.unique(sub['resolved_class']).tolist() == ['S']:
                out['type'].append('arrest' + '-S')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif np.unique(sub['resolved_class']).tolist() == ['M']:
                out['type'].append('arrest' + '-M')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            else:
                out['type'].append('normal')
                out['arrest'].append(np.nan)
                cls = np.unique(sub['resolved_class']).tolist()
                remain = ['G1', 'G2', 'S']
                for c in cls:
                    if c == 'M' or c == 'G1/G2': continue
                    l = np.sum(sub['resolved_class'] == c)
                    if sub['resolved_class'].tolist()[0] == c:
                        l = '>' + str(l)
                    elif sub['resolved_class'].tolist()[-1] == c:
                        l = '>' + str(l)
                    out[c].append(l)
                    remain.remove(c)
                for u in remain:
                    out[u].append(np.nan)
        out = pd.DataFrame(out)

        # register mitosis, mitosis time only registered in daughter 'M'
        for i in self.mt_dic.keys():
            for j in self.mt_dic[i]['daug'].keys():
                m = self.mt_dic[i]['daug'][j]['m_exit'] - self.mt_dic[i]['div']
                out.loc[out['track'] == j, 'M'] = m

        # filter length
        out = out[out['length'] >= self.minTrack]

        # imprecise M (daughter associated with parent, but daughter no M classification) exit labeled
        out['imprecise_exit'] = 0
        for i in out['track'].tolist():
            if i in self.impreciseExit:
                out.loc[out['track'] == i, 'imprecise_exit'] = 1

        return out
