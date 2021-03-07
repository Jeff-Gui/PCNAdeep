#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:49:54 2021

@author: Yifan Gui
"""
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from preparePCNA import load_PCNAs_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class PCNAEvaluator:
    
    def __init__(self, model_path, cfg_path, class_name=["G1/G2","S","M","E"]):
        self.CLASS_NAMES = class_name
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = model_path
        self.model = build_model(self.cfg)
        
    def run_evaluate(self, dataset_ann_path, dataset_path, out_dir=None):
        DatasetCatalog.register("pcna", lambda: load_PCNAs_json(dataset_ann_path, dataset_path))
        MetadataCatalog.get("pcna").set(thing_classes=self.CLASS_NAMES, evaluator_type='coco')
        
        if out_dir is None:
            evaluator = COCOEvaluator("pcna", self.cfg, False, output_dir=out_dir)
        else:
            evaluator = COCOEvaluator("pcna", self.cfg, False)
            
        val_loader = build_detection_test_loader(self.cfg, "pcna")
        inference_on_dataset(self.model, val_loader, evaluator)
        