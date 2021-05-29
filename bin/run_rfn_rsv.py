from pcnaDeep.resolver import Resolver
from pcnaDeep.refiner import Refiner
import numpy as np


def run_rfn_rsv(track, d_trh, t_trh, smooth, minG=6, minS=5, minM=3, minTrack=10):
    """Wrapper function to run Refiner and Resolver

    Args:
        track (pd.DataFrame): unresolved tracked object table
        d_trh (int): distance threshold for mitosis cell (default: 150)
        t_trh (int): temporal threshold for mitosis cell (default: 5)
        smooth (int): smoothing window on classification confidence (default: 3)
        minG (int): minimum G1/G2 phase frame length (default: 6)
        minS (int): minimum S phase frame length (default: 5)
        minM (int): minimum M phase frame length (default: 3)
        minTrack (int): minimum track length to solve,
            short tracks may not be reliable. (default: 10)

    Returns:
        track (pandas.DataFrame): resolved tracked object table
        phase (pandas.DataFrame): resolved cell cycle duration by track
    """

    myRefiner = Refiner(track, threshold_mt_F=d_trh, threshold_mt_T=t_trh, smooth=smooth, minGS=np.max((minG, minS)), minM=minM)
    ann, track, mt_dic = myRefiner.doTrackRefine()

    myResolver = Resolver(track, ann, mt_dic, minG=minG, minS=minS, minM=minM, minTrack=minTrack)
    track, phase = myResolver.doResolve()
    return track, phase
