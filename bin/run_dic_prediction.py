import argparse
import multiprocessing as mp
import time
import numpy as np
import json, os

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from pcna_predictor import VisualizationDemo, pred2json

import skimage.io as io
import skimage.measure as measure
from skimage.morphology import remove_small_objects
import pandas as pd
import torch

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
        "--mask_off",
        action="store_true"
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

    demo = VisualizationDemo(cfg, thing_class=['I','M'])

    if args.input:
        m = not args.mask_off
        gray = args.is_gray # gray: THW; non-gray: THWC
        # Input image must be uint8
        imgs = io.imread(args.input)
        print("Run on image shape: "+str(imgs.shape))
        if args.is_slice:
            imgs = np.expand_dims(imgs, axis=0)
        imgs_out = []
        table_out = pd.DataFrame()
        mask_out = []
        json_out = {}
        for i in range(imgs.shape[0]):
            img = imgs[i,:]
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
                # Generate mask or visualized output
                predictions, visualized_output = demo.run_on_image(img)
                #print(predictions['instances'].pred_classes)
                if m:
                    # Generate mask
                    mask = predictions['instances'].pred_masks
                    mask = mask.char().cpu().numpy()
                    mask_slice = np.zeros((mask.shape[1], mask.shape[2])).astype('uint8')
                
                    # For visualising class prediction
                    # 0: I, 1: M
                    cls = predictions['instances'].pred_classes
                    conf = predictions['instances'].scores
                    factor = {0:100,1:200}
                    for s in range(mask.shape[0]):
                        f = factor[cls[s].item()]
                        sc = conf[s].item()
                        ori = np.max(mask_slice[mask[s,:,:]!=0])
                        if ori!=0:
                            if sc>conf[ori-1].item():
                                mask_slice[mask[s,:,:,]!=0] = s+1
                        else:
                            mask_slice[mask[s,:,:]!=0] = s+1
                    
                    img = remove_small_objects(mask_slice,1000)
                    props = measure.regionprops_table(img, properties=('label','bbox','centroid'))
                    props = pd.DataFrame(props)

                    img_relabel = measure.label(img)
                    props_relabel = measure.regionprops_table(img_relabel, properties=('label','centroid'))
                    props_relabel = pd.DataFrame(props_relabel)
                    props_relabel.columns = ['continuous_label','Center_of_the_object_0', 'Center_of_the_object_1']

                    out_props = pd.merge(props, props_relabel, on=['Center_of_the_object_0','Center_of_the_object_1'])
                    out_props['frame'] = i
                    phase = []
                    confid = []
                    for row in range(out_props.shape[0]):
                        lb = int(out_props.iloc[row][0])
                        phase.append(cls[lb-1].item())
                        confid.append(conf[lb-1].item())
                    out_props['phase'] = phase
                    out_props['confid'] = confid
                    out_props['Center_of_the_object_0'] = np.round(out_props['Center_of_the_object_0'])
                    out_props['Center_of_the_object_1'] = np.round(out_props['Center_of_the_object_1'])
                    del out_props['label']
                    
                    table_out = table_out.append(out_props)
                    mask_out.append(img_relabel.astype('uint8'))
                else:
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
            with(open(args.output, 'w', encoding='utf8')) as file:
                json.dump(json_out, file)
        elif m:
            out = np.stack(mask_out, axis=0)
            io.imsave(args.output, out)
            table_out.to_csv(args.output[:-3]+'csv')
        else:
            out = np.stack(imgs_out, axis=0)
            io.imsave(args.output, out)
