#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from prepareKaggleNucleus import load_KaggleNucleus

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name=dataset_name, output_dir=output_folder, tasks=('segm',), distributed=True))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # 注册数据集和元数据
#声明类别，尽量保持
CLASS_NAMES =["cell"]
# 数据集路径
DATASET_ROOT = '/home/zje/dataset/kaggle_nucleus_2018'

TRAIN_PATH = os.path.join(DATASET_ROOT, 'stage1_train')
VAL_PATH = os.path.join(DATASET_ROOT, 'stage1_val')

def plain_register_dataset():
    #训练集
    DatasetCatalog.register("nucleus_train", lambda: load_KaggleNucleus(TRAIN_PATH))
    MetadataCatalog.get("nucleus_train").set(thing_classes=CLASS_NAMES, evaluator_type='coco')

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("nucleus_val", lambda: load_KaggleNucleus(VAL_PATH))
    MetadataCatalog.get("nucleus_val").set(thing_classes=CLASS_NAMES,evaluator_type='coco')

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    args.config_file = "../../configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("nucleus_train",) # 训练数据集名称
    cfg.DATASETS.TEST = ("nucleus_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程
    cfg.MODEL.MASK_ON = True 
    '''
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 256 # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 1440 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = 256 # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 1440
    '''
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    #choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = 'mrcnn_R_50_FPN_3x_mod.pkl'  # 预训练模型权重,运行configmrcnn按照类别数目修改层
    cfg.SOLVER.IMS_PER_BATCH = 4  # batch_size=4; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    ITERS_IN_ONE_EPOCH = int(570 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 4) - 1 # 4 epochs
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    
    '''
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    '''

    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    #cfg.TEST.EVAL_PERIOD = 100

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    plain_register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        )
