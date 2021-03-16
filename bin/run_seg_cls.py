import argparse
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd
import skimage.io as io
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from pcnaDeep.predictor import VisualizationDemo, pred2json


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../config/pcna_res50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="Path to image stack file.",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--is_gray",
        action="store_true"
    )
    parser.add_argument(
        "--is_slice",
        action="store_true"
    )
    parser.add_argument(
        "--json_out",
        action="store_true"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        gray = args.is_gray # gray: THW; non-gray: THWC
        # Input image must be uint8
        imgs = io.imread(args.input)
        print("Run on image shape: "+str(imgs.shape))
        if args.is_slice:
            imgs = np.expand_dims(imgs, axis=0)
        imgs_out = []
        table_out = pd.DataFrame()
        json_out = {}
        for i in range(imgs.shape[0]):
            img = imgs[i,:]
            if gray:
                img = np.stack([img, img, img], axis=2)  # convert gray to 3 channels
            start_time = time.time()
            if args.json_out:
                # Generate json output readable by VIA2
                predictions = demo.run_on_image(img, vis=False)
                ist = predictions['instances']
                labels = list(ist.pred_classes.cpu().numpy())
                masks = list(ist.pred_masks.cpu().numpy())
                file_name = os.path.basename(args.input)[:-4] + '-' + "%04d" % (i+1) + '.png'
                dic_frame = pred2json(masks, labels, file_name)
                json_out[file_name] = dic_frame
            else:
                # Generate visualized output
                predictions, visualized_output = demo.run_on_image(img)
                imgs_out.append(visualized_output.get_image())
            logger.info(
                "{}: {} in {:.2f}s".format(
                    'frame'+str(i),
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
        if args.json_out:
            prefix = re.search('(.*)\..*', os.path.basename(args.input)).group(1)
            with(open(args.output+prefix+'.json', 'w', encoding='utf8')) as file:
                json.dump(json_out, file)
        else:
            out = np.stack(imgs_out, axis=0)
            io.imsave(args.output+prefix+'_vis.tif', out)
