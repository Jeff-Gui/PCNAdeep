import argparse
import multiprocessing as mp
import time
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from pcna_predictor import VisualizationDemo

import skimage.io as io
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
        m = not args.mask_off
        gray = args.is_gray # gray: THW; non-gray: THWC
        # Input image must be uint8
        imgs = io.imread(args.input)
        # print(imgs.shape)
        if args.is_slice:
            imgs = np.expand_dims(imgs, axis=0)
        imgs_out = []
        mask_out = []
        for i in range(imgs.shape[0]):
            img = imgs[i,:]
            if gray:
                img = np.stack([img, img, img], axis=2)
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            #print(predictions['instances'].pred_classes)
            if m:
                # Generate mask
                mask = predictions['instances'].pred_masks
                mask = mask.char()
                
                # For visualising class prediction
                # 0: G1/G2, 1: S, 2: M
                cls = predictions['instances'].pred_classes
                factor = {0:50, 1:100, 2:200}
                for s in range(mask.shape[0]):
                    f = factor[cls[s].item()]
                    mask[s,:,:] = mask[s,:,:] * f
                
                mask_slice = torch.sum(mask, dim=0)
                #mask_slice_np = mask_slice.cpu().numpy().astype('uint8') # pseudo class output
                mask_slice_np = mask_slice.cpu().numpy().astype('bool') # mask output
                mask_out.append(mask_slice_np)
            logger.info(
                "{}: {} in {:.2f}s".format(
                'frame'+str(i),
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
                )
            )
            imgs_out.append(visualized_output.get_image())
        if m:
            out = np.stack(mask_out, axis=0)
            io.imsave(args.output, out)
        else:
            out = np.stack(imgs_out, axis=0)
            io.imsave(args.output, out)
