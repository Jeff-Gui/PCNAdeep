# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import re
import pprint
import numpy as np


def find_daugs(track, track_id):
    """Return list of daughters according to certain parent track ID.

    Args:
        track (pandas.DataFrame): tracked object table.
        track_id (int): track ID.
    """
    rt = list(np.unique(track.loc[track['parentTrackId'] == track_id, 'trackId']))
    if not rt:
        return []
    else:
        for trk in rt:
            rt.extend(find_daugs(track, trk))
        return rt


class Trk_obj:

    def __init__(self, track_path, frame_base=1):
        """
        To correct track ID, mitosis relationship, cell cycle classifications.

        Args:
            track_path (str): path to tracked object table.
            frame_base (int): base of counting frames, default 1.
        """

        self.track_path = track_path
        self.track = pd.read_csv(track_path)
        self.saved = None
        self.original = self.track.copy()
        self.frame_base = frame_base
        self.parser = argparse.ArgumentParser()
        self.__construct_parser()
        self.track_count = int(np.max(self.track['trackId']))
        return

    def __construct_parser(self):
        self.parser.add_argument("-t", help="Track ID.")
        self.parser.add_argument("-t1", help="Old track ID to replace.")
        self.parser.add_argument("-t2", help="New track ID to replace with.")
        self.parser.add_argument("-f", help="Time frame.")
        self.parser.add_argument("-p", help="Parent track ID.")
        self.parser.add_argument("-d", help="Daughter track ID.")
        self.parser.add_argument("-l", help="Correct classification to assign.")
        self.parser.add_argument("-s", help="Correct classification on single slice.", action='store_true')
        self.parser.add_argument("-e", help="Correct classification until the specified frame.")

        return

    def create_or_replace(self, old_id, frame, new_id=None):
        """Create a new track ID or replace with some track ID
        after certain frame. If the old track has daughters, new track ID will be the parent.

        Args:
            old_id (int): old track ID.
            frame (int): frame to begin with new ID.
            new_id (int): new track ID, only required when replacing track identity.
        """
        if old_id not in self.track['trackId']:
            raise ValueError('Selected track is not in the table.')
        if frame not in list(self.track[self.track['trackId'] == old_id]['frame']):
            raise ValueError('Selected frame is not in the original track.')

        dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == old_id, 'trackId']))
        for dd in dir_daugs:
            self.del_parent(dd)

        if new_id is None:
            self.track_count += 1
            new = self.track_count
            new_lin = new
            new_par = 0
        else:
            if new_id not in self.track['trackId']:
                raise ValueError('Selected new ID not in the table.')
            new = new_id
            new_lin = self.track[self.track['trackId'] == new_id]['lineageId'].values[0]
            new_par = self.track[self.track['trackId'] == new_id]['parentTrackId'].values[0]
        self.track.loc[(self.track['trackId'] == old_id) & (self.track['frame'] >= frame), 'trackId'] = new
        self.track.loc[self.track['trackId'] == new, 'lineageId'] = new_lin
        self.track.loc[self.track['trackId'] == new, 'parentTrackId'] = new_par
        print('Replaced/Created track ' + str(old_id) + ' from ' + str(frame+self.frame_base) +
              ' with new ID ' + str(new) + '.')

        for dd in dir_daugs:
            self.create_parent(new, dd)

        return

    def create_parent(self, par, daug):
        """Create parent-daughter relationship.

        Args:
            par (int): parent track ID.
            daug (int): daughter track ID.
        """
        if par not in self.track['trackId']:
            raise ValueError('Selected parent is not in the table.')
        if daug not in self.track['trackId']:
            raise ValueError('Selected daughter is not in the table.')

        ori_par = self.track[self.track['trackId'] == daug]['parentTrackId'].iloc[0]
        if ori_par != 0:
            raise ValueError('One daughter cannot have more than one parent, disassociate ' + str(ori_par) + '-'
                             + str(daug) + ' first.')

        par_lin = self.track[self.track['trackId'] == par]['lineageId'].iloc[0]
        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = par_lin
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = par
        self.track.loc[self.track['parentTrackId'] == daug, 'lineageId'] = par_lin
        print('New parent ' + str(par) + ' associated with daughter ' + str(daug) + '.')

        return

    def del_parent(self, daug):
        """Remove parent-daughter relationship, for a daughter.

        Args:
            daug (int): daughter track ID.
        """
        if daug not in self.track['trackId']:
            raise ValueError('Selected daughter is not in the table.')

        self.track.loc[self.track['trackId'] == daug, 'lineageId'] = daug
        self.track.loc[self.track['trackId'] == daug, 'parentTrackId'] = 0
        daugs = find_daugs(self.track, daug)
        for d in daugs:
            self.track.loc[self.track['trackId'] == d, 'lineageId'] = daug
            print('Daughter ' + str(d) + ' disassociated its original parent.')

        return

    def correct_cls(self, trk_id, frame, cls, mode='to_next', end_frame=None):
        """Correct cell cycle classification, will also influence confidence score.

        Args:
            trk_id (int): track ID to correct.
            frame (int): frame to correct or begin with correction.
            cls (str): new classification to assign.
            mode (str): either 'to_next', 'single', or 'range'
            end_frame (int): optional, in 'range' mode, stop correction at this frame.
        """
        if trk_id not in self.track['trackId']:
            raise ValueError('Selected track is not in the table.')
        if cls not in ['G1', 'G2', 'M', 'S', 'G1/G2']:
            raise ValueError('cell cycle phase can only be G1, G2, G1/G2, S or M')

        clss = list(self.track[self.track['trackId'] == trk_id]['predicted_class'])
        frames = list(self.track[self.track['trackId'] == trk_id]['frame'])
        if frame not in frames:
            raise ValueError('Selected frame is not in the original track.')
        fm_id = frames.index(frame)
        idx = self.track[self.track['trackId'] == trk_id].index
        if mode == 'single':
            rg = [fm_id]
        elif mode == 'range':
            rg = [i for i in range(fm_id, frames.index(end_frame))]
        elif mode == 'to_next':
            cur_cls = clss[fm_id]
            j = fm_id + 1
            while clss[j] == cur_cls and j < len(clss):
                j += 1
            rg = [i for i in range(fm_id, j)]
        else:
            raise ValueError('Mode can only be single, to_next or range, not ' + mode)

        for r in rg:
            self.track.loc[idx[r], 'resolved_class'] = cls
            self.track.loc[idx[r], 'predicted_class'] = cls
            if cls in ['G1', 'G2', 'G1/G2']:
                self.track.loc[idx[r], 'Probability of G1/G2'] = 1
                self.track.loc[idx[r], 'Probability of S'] = 0
                self.track.loc[idx[r], 'Probability of M'] = 0
            elif cls == 'S':
                self.track.loc[idx[r], 'Probability of G1/G2'] = 0
                self.track.loc[idx[r], 'Probability of S'] = 1
                self.track.loc[idx[r], 'Probability of M'] = 0
            else:
                self.track.loc[idx[r], 'Probability of G1/G2'] = 0
                self.track.loc[idx[r], 'Probability of S'] = 0
                self.track.loc[idx[r], 'Probability of M'] = 1
        print('Classification for track ' + str(trk_id) + ' corrected as ' + str(cls) + ' from ' +
              str(frames[rg[0]] + self.frame_base) + ' to ' + str(frames[rg[-1]] + self.frame_base) + '.')

        return

    def delete_track(self, trk_id):
        """Delete entire track.

        Args:
            trk_id (int): track ID.
        """

        # For all direct daughter of the track to delete, first remove association
        dir_daugs = list(np.unique(self.track.loc[self.track['parentTrackId'] == trk_id, 'trackId']))
        for dd in dir_daugs:
            self.del_parent(dd)

        # Delete entire track
        self.track.drop(index=self.track[self.track['trackId'] == trk_id].index)
        return

    def save(self):
        """Save current table.
        """
        self.getAnn()
        self.track.to_csv(self.track_path, index=None)
        self.saved = self.track.copy()
        return

    def revert(self):
        """Revert to last saved version.
        """
        if self.saved is None:
            raise ValueError('Please save last changes first before reverting.')
        self.track = self.saved.copy()
        return

    def erase(self):
        """Erase all editing to the original file.
        """
        self.track = self.original.copy()
        return

    def getAnn(self):
        """Add an annotation column to tracked object table
        The annotation format is track ID - (parentTrackId, optional) - resolved_class
        """
        ann = []
        for i in range(self.track.shape[0]):
            inform = list(self.track.iloc[i][['trackId', 'parentTrackId', 'resolved_class']])
            inform = list(map(lambda x:str(x), inform))
            if inform[1] == '0':
                del inform[1]
            ann.append('-'.join(inform))
        self.track['name'] = ann
        return

    def doCorrect(self):
        """Iteration for user command input.
        """
        while True:
            ipt = input("@ Correct > ")
            ipt_list = re.split('\s+', ipt)

            cmd = ipt_list[0]
            #  print(ipt_list[1:])
            args = self.parser.parse_args(ipt_list[1:])
            if args.f:
                args.f = int(args.f) - self.frame_base

            try:
                if cmd == 'cls':
                    md = 'to_next'
                    if args.s:
                        md = 'single'
                    elif args.e:
                        md = 'range'
                        args.e = int(args.e)
                    self.correct_cls(int(args.t), args.f, str(args.l), md, end_frame=args.e)
                elif cmd == 'r':
                    self.create_or_replace(int(args.t1), args.f, int(args.t2))
                elif cmd == 'c':
                    self.create_or_replace(int(args.t), args.f)
                elif cmd == 'cp':
                    self.create_parent(int(args.p), int(args.d))
                elif cmd == 'dp':
                    self.del_parent(int(args.d))
                elif cmd == 'del':
                    self.delete_track(int(args.t))
                elif cmd == 's':
                    self.save()
                elif cmd == 'q':
                    break
                elif cmd == 'wq':
                    self.save()
                    break
                elif cmd == 'revert':
                    self.revert()
                elif cmd == 'erase':
                    self.erase()
                else:
                    print("Wrong command argument!")
                    print("=================== Available Commands ===================\n")
                    pprint.pprint({'cls -t  -f  -l (-s/-e)':'Correct cell cycle classification for track (t) from frame'
                                                            ' (f) with class label (l).',
                                   'r   -t1 -t2 -f        ':'Replace track ID (t1) with a new one (t2) from frame (f).',
                                   'c   -t  -f            ':'Create new track ID for track (t) from frame (f)',
                                   'cp  -p  -d            ':'Create parent (p) - daughter (d) relationship',
                                   'dp  -d                ':'Delete parent - daughter (d) relationship',
                                   'del -t                ':'Delete entire track',
                                   'q                     ':'Quit the interface',
                                   's                     ':'Save the current table',
                                   'wq                    ':'Save and quit the interface',
                                   'revert                ':'Revert to last saved version',
                                   'erase                 ':'Erase to original version'})

                    print("\n=================== Parameter Usages ====================\n")
                    self.parser.print_help()
            except ValueError as v:
                print(repr(v))
                continue

        return


'''
Testing inputs
cls -t 1 -f 10 -l S

'''
