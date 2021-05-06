# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from pcnaDeep.refiner import deduce_transition
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


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
        if a[i] == 'G1/G2' and (b[i] in ['G1', 'G2', 'G1*', 'G2*']):
            count -= 1

    return count


class Resolver:

    def __init__(self, track, ann, mt_dic, minG=6, minS=5, minM=3, minTrack=10, impreciseExit=None, G2_trh=100):
        if impreciseExit is None:
            impreciseExit = []
        self.logger = logging.getLogger('pcna.Resolver')
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
        self.mt_unresolved = []
        self.arrest = {}  # trackId : arrest phase
        self.G2_trh = G2_trh
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
        if self.mt_unresolved:
            self.logger.info('Sequential mitosis without S phase; Ignore tracks: ' + str(self.mt_unresolved)[1:-1])
        if self.unresolved:
            self.logger.info('Numerous classification change after resolving, check: ' + str(self.unresolved)[1:-1])

        self.resolveArrest(self.G2_trh)
        phase = self.doResolvePhase()
        return self.rsTrack, phase

    def resolveArrest(self, G2_trh=None):
        """Determine G1/G2 arrest tracks;
        Assign G1 or G2 classification according to 2-center K-mean.

        Args:
            G2_trh (int): int between 1-255, above the threshold will be classified as G2
        """

        trk = self.rsTrack[self.rsTrack['trackId'].isin(list(self.arrest.keys()))].copy()
        intensity = []
        ids = []
        for i in self.arrest.keys():
            sub = trk[trk['trackId'] == i]
            corrected_mean = np.mean(sub['mean_intensity'] - sub['background_mean'])
            intensity.append(corrected_mean)
            ids.append(i)

        #  print(intensity)
        if G2_trh is None:
            X = np.expand_dims(np.array(intensity), axis=1)
            X = MinMaxScaler().fit_transform(X)
            y = list(KMeans(2).fit_predict(X))
        else:
            if G2_trh < 1 or G2_trh > 255:
                raise ValueError('G2 threshold must be within the interval: 1~255.')
            y = []
            for i in range(len(ids)):
                if intensity[i] > G2_trh:
                    y.append(1)
                else:
                    y.append(0)

        for i in range(len(ids)):
            if y[i] == 0:
                self.arrest[ids[i]] = 'G1'
                self.rsTrack.loc[self.rsTrack['trackId'] == ids[i], 'resolved_class'] = 'G1*'
            else:
                self.arrest[ids[i]] = 'G2'
                self.rsTrack.loc[self.rsTrack['trackId'] == ids[i], 'resolved_class'] = 'G2*'
        return

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
        resolved_class = ['G1/G2' for _ in range(trk.shape[0])]
        if trk.shape[0] == 0:
            raise ValueError('Track not found!')

        track_id = trk['trackId'].tolist()[0]
        cls = list(trk['predicted_class'])
        confid = np.array(trk[['Probability of G1/G2', 'Probability of S', 'Probability of M']])
        out = deduce_transition(l=cls, tar='S', confidence=confid, min_tar=self.minS,
                                max_res=np.max((self.minM, self.minG)), casual_end=False)

        flag = False
        if not (out is None or out[0] == out[1]):
            flag = True
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

        if not flag and m_exit is not None and m_entry is not None:
            resolved_class = cls.copy()
            self.mt_unresolved.append(track_id)

        if m_exit is None and m_entry is None:
            # some tracks begin/end with mitosis and not associated during refinement. In this case, override any
            # classification at terminal
            # WARNING: this can leads to false negative
            mt_out_begin = deduce_transition(l=cls, tar='M', confidence=confid, min_tar=1,
                                             max_res=np.max((self.minS, self.minG)))
            mt_out_end = deduce_transition(l=cls[::-1], tar='M', confidence=confid[::-1, :], min_tar=1,
                                           max_res=np.max((self.minS, self.minG)))
            if mt_out_end is not None:
                #  check if out and end interval overlaps
                compare = (len(cls)-mt_out_end[1]-1, len(cls)-mt_out_end[0]-1)
                if mt_out_begin is not None:
                    if compare[0] == mt_out_begin[0] and compare[1] == mt_out_begin[1]:
                        #  if overlap, assign larger index one to None
                        if mt_out_end[0] < mt_out_begin[1]:
                            mt_out_begin = None
                        else:
                            mt_out_end = None

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
        if np.unique(resolved_class).tolist() == ['G1/G2']:
            self.arrest[track_id] = 'G1'
        if list_dist(cls, resolved_class) > UNRESOLVED_FRACTION * len(resolved_class) and \
                len(resolved_class) >= self.minTrack:
            self.unresolved.append(track_id)
        return trk

    def doResolvePhase(self):
        """Resolve phase durations
        """
        out = {'track': [], 'type': [], 'length': [], 'arrest': [], 'G1': [], 'S': [], 'M': [], 'G2': [], 'parent': []}

        # register tracks
        for i in range(self.ann.shape[0]):
            info = self.ann.loc[i, :]
            if info['track'] in self.unresolved or info['track'] in self.mt_unresolved:
                continue
            sub = self.rsTrack[self.rsTrack['trackId'] == info['track']]
            length = np.max(sub['frame']) - np.min(sub['frame'])
            par = info['mitosis_parent']
            if par is None or par in self.mt_unresolved:
                par = 0
            out['track'].append(info['track'])
            out['length'].append(length)
            out['parent'].append(par)
            out['M'].append(np.nan)  # resolve later

            if list(np.unique(sub['resolved_class'])) == ['G1*']:
                out['type'].append('arrest' + '-G1')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['G2*']:
                out['type'].append('arrest' + '-G2')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['S']:
                out['type'].append('arrest' + '-S')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            elif list(np.unique(sub['resolved_class'])) == ['M']:
                out['type'].append('arrest' + '-M')
                out['arrest'].append(length)
                out['G1'].append(np.nan)
                out['S'].append(np.nan)
                out['G2'].append(np.nan)
            else:
                out['type'].append('normal')
                out['arrest'].append(np.nan)
                cls = list(np.unique(sub['resolved_class']))
                remain = ['G1', 'G2', 'S']
                for c in cls:
                    if c == 'M' or c == 'G1/G2':
                        continue
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

        # register mitosis, mitosis time only registered in daughter 'M' column
        for i in self.mt_dic.keys():
            if i in self.mt_unresolved:
                continue
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
