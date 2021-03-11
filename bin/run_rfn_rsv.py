from pcnaDeep.data.resolver import Resolver
from pcnaDeep.data.refiner import Refiner

myRefiner = Refiner(track, threshold_mt_F=150, threshold_mt_T=5, smooth=5, minGS=3, minM=3)
ann, track, mt_dic = myRefiner.doTrackRefine()

myResolver = Resolver(track, ann, mt_dic, minG=5, minS=6, minM=3, minTrack=10)
track, phase = myResolver.doResolve()