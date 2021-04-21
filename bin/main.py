import argparse
import multiprocessing as mp
import os
import re
import time
import yaml

import numpy as np
import pandas as pd
import skimage.io as io
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from pcnaDeep.predictor import VisualizationDemo, predictFrame
from pcnaDeep.refiner import Refiner
from pcnaDeep.resolver import Resolver
from pcnaDeep.tracker import track
from pcnaDeep.split import split_frame, join_frame, join_table, resolve_joined_stack
from pcnaDeep.utils import getDetectInput


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.dtrn_config)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="pcnaDeep configs.")
    parser.add_argument(
        "--dtrn-config",
        default="../config/dtrnCfg.yaml",
        metavar="FILE",
        help="path to detectron2 model config file",
    )
    parser.add_argument(
        "--pcna-config",
        default="../config/pcnaCfg.yaml",
        metavar="FILE",
        help="path to pcnaDeep tracker/refiner/resolver config file",
    )
    parser.add_argument(
        "--pcna",
        default=None,
        help="Path to PCNA channel time series image.",
    )
    parser.add_argument(
        "--dic",
        default=None,
        help="Path to DIC channel time series image.",
    )
    parser.add_argument(
        "--stack-input",
        default=None,
        help="Path to composite image stack file. Will overwrite pcna or dic input, not recommended.",
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
        "--opts",
        help="Modify pcnaDeep config options using the command-line 'KEY VALUE' pairs. For pcnaDeep config, "
             "begin with pcna., e.g., pcna.TRACKER.DISPLACE 100. For detectron2 config, follow detectron2 docs",
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
    # resolve pcnaDeep Config
    with open(args.pcna_config, 'rb') as f:
        date = yaml.safe_load_all(f)
        pcna_cfg_dict = list(date)[0]
    dtrn_opts = []
    i = 0
    while i < len(args.opts)/2:
        o = args.opts[2*i]
        value = args.opts[2*i+1]
        l = o.split('.')
        if l[0] == 'pcna' or l[0] == 'PCNA':
            if len(l) == 2:
                pcna_cfg_dict[l[1]] = value
            elif len(l) >= 3:
                cur_ref = pcna_cfg_dict[l[1]]
                for j in range(2, len(l)-1):
                    cur_ref = cur_ref[l[j]]
                cur_ref[l[-1]] = value
        else:
            dtrn_opts.append(o)
            dtrn_opts.append(value)
        i += 1
    args.opts = dtrn_opts
    cfg = setup_cfg(args)
    logger.info("Finished setup.")
    demo = VisualizationDemo(cfg)

    logger.info("Start inferring.")
    ipt = args.stack_input
    if (ipt is not None or (args.pcna is not None and args.dic is not None)) and not pcna_cfg_dict['BATCH']:
        flag = True
        if ipt is not None:
            prefix = os.path.basename(ipt)
            prefix = re.match('(.+)\.\w+',prefix).group(1)
            # Input image must be uint8
            imgs = io.imread(ipt)
        else:
            prefix = os.path.basename(args.pcna)
            prefix = re.match('(.+)\.\w+', prefix).group(1)
            pcna = io.imread(args.pcna)
            dic = io.imread(args.dic)
            logger.info("Generating composite...")
            imgs = getDetectInput(pcna, dic)
            del pcna
            del dic

        logger.info("Run on image shape: " + str(imgs.shape))
        table_out = pd.DataFrame()
        mask_out = []
        spl = int(pcna_cfg_dict['SPLIT']['GRID'])
        if spl:
            new_imgs = []
            for i in range(imgs.shape[0]):
                splited = split_frame(imgs[i,:].copy(), n=spl)
                for j in range(splited.shape[0]):
                    new_imgs.append(splited[j,:])
            imgs = np.stack(new_imgs, axis=0)
            del new_imgs

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
        
        tw = imgs.shape[1]
        del imgs  # save memory space TODO: use image buffer input
        
        mask_out = np.stack(mask_out, axis=0)

        if spl:
            mask_out = join_frame(mask_out.copy(), n=spl)
            table_out = join_table(table_out.copy(), n=spl, tile_width=tw)
            mask_out, table_out = resolve_joined_stack(mask_out, table_out, n=spl, 
                                                       boundary_width=pcna_cfg_dict['SPLIT']['EDGE_SPLIT'],
                                                       dilate_time=pcna_cfg_dict['SPLIT']['DILATE_ROUND'])
        
        logger.info('Tracking...')
        track_out = track(df=table_out, displace=int(pcna_cfg_dict['TRACKER']['DISPLACE']),
                          gap_fill=int(pcna_cfg_dict['TRACKER']['GAP_FILL']))
        track_out.to_csv(os.path.join(args.output, prefix + '_tracks.csv'), index=0)
        io.imsave(os.path.join(args.output, prefix + '_mask.tif'), mask_out)

        logger.info('Refining and Resolving...')
        post_cfg = pcna_cfg_dict['POST_PROCESS']
        refiner_cfg = post_cfg['REFINER']
        myRefiner = Refiner(track_out, threshold_mt_F=int(refiner_cfg['MAX_DIST_TRH']),
                            threshold_mt_T=int(refiner_cfg['MAX_FRAME_TRH']), smooth=int(refiner_cfg['SMOOTH']),
                            minGS=np.max((int(post_cfg['MIN_G']), int(post_cfg['MIN_S']))),
                            minM=int(post_cfg['MIN_M']), search_range=int(refiner_cfg['SEARCH_RANGE']),
                            mt_len=int(refiner_cfg['MITOSIS_LEN']), sample_freq=float(refiner_cfg['SAMPLE_FREQ']),
                            model_train=refiner_cfg['SVM_TRAIN_DATA'],
                            mode=refiner_cfg['MODE'])
        ann, track_rfd, mt_dic, imprecise = myRefiner.doTrackRefine()

        myResolver = Resolver(track_rfd, ann, mt_dic, minG=int(post_cfg['MIN_G']), minS=int(post_cfg['MIN_S']),
                              minM=int(post_cfg['MIN_M']), 
                              minTrack=int(post_cfg['RESOLVER']['MIN_TRACK']), impreciseExit=imprecise)
        track_rsd, phase = myResolver.doResolve()
        track_rsd.to_csv(os.path.join(args.output, prefix + '_tracks_refined.csv'), index=0)
        phase.to_csv(os.path.join(args.output, prefix + '_phase.csv'), index=0)

        logger.info('Finished: '+time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
