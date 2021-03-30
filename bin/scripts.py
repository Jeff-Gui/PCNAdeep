import skimage.io as io
import pandas as pd
import os
import numpy as np
json_fp = '/Users/jefft/Downloads/20200902_s2_cpd/20200902_MCF10A-s2_cpd.json'
raw_img_root = '/Users/jefft/Downloads/20200902_s2_cpd/'
out_fp = '/Users/jefft/Downloads/20200902_s2_cpd/'

#%% Generate trk file for annotation

from pcnaDeep.data.utils import json2mask
mask = json2mask(ip=json_fp, out='', height=1200, width=1200, 
          label_phase=True, mask_only=True)  # suppress file output by mask_only=True, record phase by label_phase=True

from pcnaDeep.tracker import track_mask
# instead of input mask labeled with cell-cycle specific value, the output mask is continuously labeled
tracked, labeled_mask = track_mask(mask, render_phase=True, displace=100, gap_fill=3)

imgs = []
imgs_fp = os.listdir(raw_img_root)
imgs_fp.sort()
for i in imgs_fp:
    if i[-3:] == 'png':
        img = io.imread(os.path.join(raw_img_root, i))
        imgs.append(img)
raw = np.stack(imgs, axis=0)

from pcnaDeep.data.annotate import generate_calibanTrk
tracked_new = generate_calibanTrk(raw=raw, mask=labeled_mask, out_dir=out_fp, 
                    dt_id=2, digit_num=4, track=tracked)
tracked_new.to_csv(os.path.join(out_fp, '0002_tracked.csv'), index=0)

#%%
# Correct .trk in deepcell-label

#%% Get tracked ground truth
from pcnaDeep.data.annotate import mergeTrkAndTrack
out = mergeTrkAndTrack(trk_path = os.path.join(out_fp, '0002.trk'),
                      table_path = os.path.join(out_fp, '0002_tracked.csv'),
                      return_mask = False)
track = out[0]
track.to_csv(os.path.join(out_fp, '0002_tracked_GT.csv'), index=0)
mt_dic = out[1]

#%% Generate mt lookup
from pcnaDeep.data.utils import mt_dic2mt_lookup
mt_lookup = mt_dic2mt_lookup(mt_dic)
#   optional: save or not
# mt_lookup.to_csv(os.path.join(out_fp, '0002_mitosis_lookup.txt'), index=0)

#%% Generate feature map
from pcnaDeep.refiner import Refiner
r = Refiner(track = pd.read_csv(os.path.join(out_fp, '0002_tracked_GT.csv')), mode='TRAIN',
           sample_freq=20, mt_len=5) # remember to input metadata: sample frequency and mitosi length
X, y = r.get_SVM_train(np.array(mt_lookup), random_negative=True, rand_size=500)
X = pd.DataFrame(np.array(X))
X['5'] = y

#%% Merge from old training features and save new training set
base = pd.read_csv('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/models/SVM_train.txt', header=None)
merged = np.concatenate((np.array(X), np.array(base)), axis=0)
pd.DataFrame(merged).to_csv('/Users/jefft/Desktop/Chan lab/SRTP/ImageAnalysis/PCNAdeep/models/SVM_train_new.txt', index=0, header=None)

#%% Prepare new model
X = merged[:,:5]
y = merged[:,5]
# normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

import matplotlib.pyplot as plt
sub = X[:,[0,1,2]]  # spatial and temporal distance and parent mitosis score, the first three features
plt.scatter(sub[:,0], sub[:,1], c=y, s=(- min(sub[:,2]) + sub[:,2])*10, alpha=0.8, cmap='coolwarm')

#%% Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

model = SVC(probability=True, class_weight='balanced')
scores = cross_val_score(model, X, y, cv=5)
print(scores)

#%% Model evaluation
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
pre_y_list = model.fit(X, y).predict(X)
tmp_list = []
for m in model_metrics_name:
    tmp_score = m(y, pre_y_list)
    tmp_list.append(tmp_score)
df2 = pd.DataFrame([tmp_list], index=['SVM'], columns=['ev', 'mae', 'mse', 'r2'])
print(df2)
# refs：https://blog.csdn.net/qq_41076797/article/details/101037721

#%% Save model
model.fit(X, y)
import joblib
joblib.dump(model, os.path.join(out_fp, 'mitosis_svm.m'))


