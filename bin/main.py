import multiprocessing as mp
import numpy as np
import os, re, time, argparse

import skimage.io as io
import pandas as pd
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from pcnaDeep.predictor import VisualizationDemo, predictFrame
from pcnaDeep.tracker import track
from pcnaDeep.resolver import Resolver
from pcnaDeep.refiner import Refiner


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
        default="../config/pcna_res50.yaml",
        metavar="FILE",
        help="path to detectron2 model config file",
    )
    parser.add_argument(
        "--input",
        help="Path to image stack file.",
    )
    parser.add_argument(
        "--output",
        help="Output directory",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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
        "--minG",
        default=6,
        help='Refinement/Resolver: minimum G1/G2 duration',
    )
    parser.add_argument(
        "--minS",
        default=5,
        help='Refinement/Resolver: minimum S duration',
    )
    parser.add_argument(
        "--minM",
        default=3,
        help='Refinement/Resolver: minimum M duration',
    )
    parser.add_argument(
        "--d_trh",
        default=150,
        help='Refinement: maximum mitosis distance tolerance',
    )
    parser.add_argument(
        "--t_trh",
        default=5,
        help='Refinement: maximum mitosis frame tolerance',
    )
    parser.add_argument(
        "--smooth",
        default=5,
        help='Refinement: smoothing window of classification',
    )
    parser.add_argument(
        "--minTrack",
        default=10,
        help='Resolver: minimum track length to report cell cycle duration',
    )
    parser.add_argument(
        "--batch",
        action="store_true"
    )
    parser.add_argument(
        "--opts",
        help="Modify detectron2 config options using the command-line 'KEY VALUE' pairs",
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
    
    logger.info("Start inferring.")
    if args.input and not args.batch:
        prefix = os.path.basename(args.input)
        prefix = re.match('(.+)\.\w+',prefix).group(1)
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
        
        del imgs  # save memory space TODO: use image buffer input
        mask_out = np.stack(mask_out, axis=0)

        logger.info('Tracking...')
        track_out = track(df=table_out, displace=int(args.displace), gap_fill=int(args.gap_fill))
        track_out.to_csv(os.path.join(args.output, prefix + '_tracks.csv'), index=0)
        io.imsave(os.path.join(args.output, prefix + '_mask.tif'), mask_out)

        logger.info('Refining and Resolving...')
        myRefiner = Refiner(track_out, threshold_mt_F=args.d_trh, threshold_mt_T=args.t_trh, smooth=args.smooth,
                            minGS=np.max((args.minG, args.minS)), minM=args.minM)
        ann, track_rfd, mt_dic = myRefiner.doTrackRefine()

        myResolver = Resolver(track_rfd, ann, mt_dic, minG=args.minG, minS=args.minS, minM=args.minM, minTrack=args.minTrack)
        track_rsd, phase = myResolver.doResolve()
        track_rsd.to_csv(os.path.join(args.output, prefix + '_tracks_refined.csv'), index=0)
        phase.to_csv(os.path.join(args.output, prefix + '_phase.csv'), index=0)

        logger.info('Finished: '+time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
