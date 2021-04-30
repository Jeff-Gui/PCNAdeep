import argparse
import multiprocessing as mp
import os
import re
import time
import yaml
import pprint
import gc

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
from pcnaDeep.data.utils import getDetectInput


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


def main(stack, config, output, prefix, logger):
    logger.info("Run on image shape: " + str(stack.shape))
    table_out = pd.DataFrame()
    mask_out = []
    spl = int(config['SPLIT']['GRID'])
    if spl:
        new_imgs = []
        for i in range(stack.shape[0]):
            splited = split_frame(stack[i,:].copy(), n=spl)
            for j in range(splited.shape[0]):
                new_imgs.append(splited[j,:])
        stack = np.stack(new_imgs, axis=0)
        del new_imgs

    for i in range(stack.shape[0]):
        start_time = time.time()
        img_relabel, out_props = predictFrame(stack[i,:], i, demo)
        table_out = table_out.append(out_props)
        mask_out.append(img_relabel)
        
        logger.info(
            "{}: {} in {:.2f}s".format(
                'frame'+str(i),
                "detected {} instances".format(out_props.shape[0]),
                time.time() - start_time,
            )
        )
    
    tw = stack.shape[1]
    del stack  # save memory space TODO: use image buffer input
    gc.collect()
    
    mask_out = np.stack(mask_out, axis=0)

    if spl:
        mask_out = join_frame(mask_out.copy(), n=spl)
        table_out = join_table(table_out.copy(), n=spl, tile_width=tw)
        mask_out, table_out = resolve_joined_stack(mask_out, table_out, n=spl, 
                                                    boundary_width=config['SPLIT']['EDGE_SPLIT'],
                                                    dilate_time=config['SPLIT']['DILATE_ROUND'])
    
    logger.info('Tracking...')
    track_out = track(df=table_out, displace=int(config['TRACKER']['DISPLACE']),
                        gap_fill=int(config['TRACKER']['GAP_FILL']))
    track_out.to_csv(os.path.join(output, prefix + '_tracks.csv'), index=0)
    #io.imsave(os.path.join(output, prefix + '_mask.tif'), mask_out)

    logger.info('Refining and Resolving...')
    post_cfg = config['POST_PROCESS']
    refiner_cfg = post_cfg['REFINER']
    myRefiner = Refiner(track_out, threshold_mt_F=int(refiner_cfg['MAX_DIST_TRH']),
                        threshold_mt_T=int(refiner_cfg['MAX_FRAME_TRH']), smooth=int(refiner_cfg['SMOOTH']),
                        minGS=np.max((int(post_cfg['MIN_G']), int(post_cfg['MIN_S']))),
                        minM=int(post_cfg['MIN_M']), search_range=int(refiner_cfg['SEARCH_RANGE']),
                        mt_len=int(refiner_cfg['MITOSIS_LEN']), sample_freq=float(refiner_cfg['SAMPLE_FREQ']),
                        model_train=refiner_cfg['SVM_TRAIN_DATA'],
                        mode=refiner_cfg['MODE'])
    ann, track_rfd, mt_dic, imprecise = myRefiner.doTrackRefine()
    
    ann.to_csv(os.path.join(output, prefix + '_tracks_ann.csv'), index=0)
    pprint.pprint(mt_dic, indent=4)

    myResolver = Resolver(track_rfd, ann, mt_dic, minG=int(post_cfg['MIN_G']), minS=int(post_cfg['MIN_S']),
                            minM=int(post_cfg['MIN_M']), 
                            minTrack=int(post_cfg['RESOLVER']['MIN_TRACK']), impreciseExit=imprecise,
                            G2_trh=int(post_cfg['RESOLVER']['G2_TRH']))
    track_rsd, phase = myResolver.doResolve()
    track_rsd.to_csv(os.path.join(output, prefix + '_tracks_refined.csv'), index=0)
    phase.to_csv(os.path.join(output, prefix + '_phase.csv'), index=0)

    logger.info(prefix+' Finished: '+time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
    logger.info('='*50)

    return


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

    if pcna_cfg_dict['BATCH']:
        ### Under construction
        if args.pcna is not None and args.dic is not None:
            if os.path.isdir(args.pcna) and os.path.isdir(args.dic) and os.path.isdir(args.output):
                pcna_imgs = os.listdir(args.pcna)
                dic_imgs = os.listdir(args.dic)
                pairs = []
                for pi in pcna_imgs:
                    prefix = os.path.basename(pi)
                    mat_obj = re.match('(.+)pcna\.tif',prefix)
                    if mat_obj is None:
                        raise ValueError('PCNA file ' + pi + ' must ends with \"pcna\" and in .tif format')
                    prefix = mat_obj.group(1)
                    if prefix + 'dic.tif' not in dic_imgs:
                        raise ValueError('DIC file ' + prefix + 'dic.tif does not exit.')
                    pairs.append((prefix, prefix + 'pcna.tif', prefix + 'dic.tif'))
                
                for si in pairs:
                    os.mkdir(os.path.join(args.output, si[0]))
                    imgs = getDetectInput(io.imread(os.path.join(args.pcna, si[1])), 
                                          io.imread(os.path.join(args.dic, si[2])))

                    inspect = imgs[range(0, imgs.shape[0], 100),:,:,:].copy()
                    io.imsave(os.path.join(args.output, si[0], si[0] + '_sample_intput.tif'), inspect)
    
                    main(stack=imgs, config=pcna_cfg_dict, output=os.path.join(args.output, si[0]), 
                         prefix=si[0], logger=logger)
            else:
                raise ValueError('Must input directory in batch mode, not single file.')
        
        elif ipt is not None:
            if os.path.isdir(ipt):
                stack_imgs = os.listdir(ipt)
                for si in stack_imgs:
                    prefix = os.path.basename(si)
                    prefix = re.match('(.+)\.\w+',si).group(1)
                    os.mkdir(os.path.join(args.output, prefix))
                    imgs = io.imread(os.path.join(ipt, si))

                    inspect = imgs[range(0, imgs.shape[0], 100),:,:,:].copy()
                    io.imsave(os.path.join(args.output, prefix, prefix + '_sample_intput.tif'), inspect)

                    main(stack=imgs, config=pcna_cfg_dict, output=os.path.join(args.output, prefix), 
                         prefix=prefix, logger=logger)
            else:
                raise ValueError('Must input directory in batch mode, not single file.')

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
            gc.collect()
        
        inspect = imgs[range(0, imgs.shape[0], 100),:,:,:].copy()
        io.imsave(args.output + prefix + '_sample_intput.tif', inspect)

        main(stack=imgs, config=pcna_cfg_dict, output=args.output, prefix=prefix, logger=logger)
        
