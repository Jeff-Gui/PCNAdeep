import pandas as pd
import numpy as np
import skimage.io as io
from data.annotate import relabel_trackID, label_by_track, get_lineage_dict, get_lineage_txt, save_trks, load_trks, lineage_dic2txt, break_track
from tracker import track_mask

# 2021/3/4
# 1. From detection and tracking output, generate RES folder files
mask = io.imread('../examples/10A_20200902_s1_cpd_trackPy/mask.tif')
mask.dtype
track = pd.read_csv('../examples/10A_20200902_s1_cpd_trackPy/output/tracks-refined.csv')
track
track_new = relabel_trackID(track.copy())
track_new, rel = break_track(track_new.copy())
tracked_mask = label_by_track(mask.copy(), track_new.copy())
txt = get_lineage_txt(track_new)
# write out processed files for RES folder
io.imsave('/Users/jefft/Desktop/mask_tracked.tif', tracked_mask.astype('uint16'))
txt.to_csv('/Users/jefft/Desktop/res_track.txt', sep=' ', index=0, header=False)

# 2. From ground truth mask, generate Caliban files for annotating tracks, eventually for GT folder files
# Ground truth mask may be annotated by VIA2
mask = io.imread('/Users/jefft/Desktop/mask_GT.tif')
raw = io.imread('/Users/jefft/Desktop/raw.tif')
out = track_mask(mask, discharge=100, gap_fill=3)
track_new = relabel_trackID(out.copy())
track_new, rel = break_track(track_new.copy())
tracked_mask = label_by_track(mask.copy(), track_new.copy())
dic = get_lineage_dict(track_new.copy(), rel)
save_trks('/Users/jefft/Desktop/001.trk', dic, np.expand_dims(raw, axis=3), np.expand_dims(tracked_mask, axis=3))

# 3. After editing in caliban, restore tracks and the mask
t = load_trks('/Users/jefft/Desktop/001-pcd.trk')
lin = t['lineages']
mask = t['y']
txt = lineage_dic2txt(lin)
# save
txt.to_csv('/Users/jefft/Desktop/man_track.txt', index=0, sep=' ', header=False)
io.imsave('/Users/jefft/Desktop/GT.tif',mask[:,:,:,0])