import argparse
import multiprocessing as mp
import time
import numpy as np
import json, os

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo, predictFrame

import skimage.io as io
import skimage.measure as measure
from skimage.morphology import remove_small_objects
import pandas as pd
from tracker import track

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
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
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
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
        "--displace",
        default=40,
        help='Tracking: maximum displacement of objects between frames',
    )
    parser.add_argument(
        "--gap_fill",
        default=5,
        help='Tracking: memory frames of objects to fill gaps',
    )
    parser.add_argument(
        "--batch",
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
    setup_logger(name="pcna")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    logger.info("Finished setup.")

    demo = VisualizationDemo(cfg)
    
    logger.info("Start infering.")
    if args.input and not args.batch:
        gray = args.is_gray # gray: THW; non-gray: THWC
        # Input image must be uint8
        imgs = io.imread(args.input)
        logger.info("Run on image shape: "+str(imgs.shape))

        table_out = pd.DataFrame()
        mask_out = []

        for i in range(imgs.shape[0]):
            start_time = time.time()
            img_relabel, out_props = predictFrame(imgs[i,:], i, demo)
            table_out = table_out.append(out_props)
            mask_out.append(img_relabel)
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                'frame'+str(i),
                "detected {} instances".format(out_props.shape[0]),
                time.time() - start_time,
                )
            )
        
        del(imgs)  # save memory space TODO: use image buffer input
        mask_out = np.stack(mask_out, axis=0)

        logger.info('Tracking...')
        track_out = track(df=table_out, discharge=int(args.displace), gap_fill=int(args.gap_fill))
        track_out[0].to_csv(os.path.join(args.output,'tracks.csv'), index=0)
        io.imsave(os.path.join(args.output,'mask.tif'), mask_out)

        logger.info('Finished: '+time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
