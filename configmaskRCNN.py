import torch
import numpy as np
import pickle
import sys, getopt

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:o:c:", ["indir=", "outdir=", "class_num="])
        # h: switch-type parameter, help
        # i: / o: parameter must with some values
    except getopt.GetoptError:
        print('deepcell_predict.py -i <input model> -o <output model>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
           print('deepcell_predict.py -i <input model> -o <output model>')
           sys.exit()
        elif opt in ("-i", "--indir"):
           ip = arg
        elif opt in ("-o", "--outdir"):
           out = arg
        elif opt in ("-c", "--class_num"):
            num_class = int(arg)

    with open(ip, 'rb') as f:
        obj = f.read()
    weights = pickle.loads(obj, encoding='latin1')

    weights['model']['roi_heads.box_predictor.cls_score.weight']=np.zeros([num_class+1,1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.cls_score.bias']=np.zeros([num_class+1], dtype='float32')

    weights['model']['roi_heads.box_predictor.bbox_pred.weight']=np.zeros([num_class*4,1024], dtype='float32')
    weights['model']['roi_heads.box_predictor.bbox_pred.bias']=np.zeros([num_class*4], dtype='float32')

    weights['model']['roi_heads.mask_head.predictor.weight']=np.zeros([num_class,256,1,1], dtype='float32')
    weights['model']['roi_heads.mask_head.predictor.bias']=np.zeros([num_class], dtype='float32')

    f = open(out, 'wb')
    pickle.dump(weights, f)
    f.close()
