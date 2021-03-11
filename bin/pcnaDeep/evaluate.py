# -*- coding: utf-8 -*-
import os, re
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from pcnaDeep.data.preparePCNA import load_PCNAs_json
from pcnaDeep.data.annotate import relabel_trackID, label_by_track, get_lineage_dict, get_lineage_txt, load_trks, lineage_dic2txt, break_track, save_seq
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model

class pcna_detectronEvaluator:
    
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

class pcna_ctcEvaluator:

    def __init__(self, root, dt_id, digit_num=3, t_base=0, path_ctc_software=None):
        """Evaluation of tracking output
        """
        self.dt_id = dt_id
        self.digit_num = digit_num
        self.t_base = t_base
        self.root = root
        self.path_ctc_software = path_ctc_software

    def generate_ctcRES(self, mask, track):
        """Generate RES format for Cell Tracking Challenge Evaluation
        """
        track_new = relabel_trackID(track.copy())
        track_new, rel = break_track(track_new.copy())
        tracked_mask = label_by_track(mask.copy(), track_new.copy())
        txt = get_lineage_txt(track_new)
        # write out processed files for RES folder
        fm = ("%0" + str(self.digit_num) + "d") % (self.dt_id)
        save_seq(tracked_mask, os.path.join(self.root, fm+'_RES'), 'mask', base=self.t_base)
        txt.to_csv(os.path.join(self.root, fm+'_RES', 'res_track.txt'), sep=' ', index=0, header=False)
        return

    def caliban2ctcGT(self, trk_path, dt_id=None, digit_num=3):
        """Convert caliban ground truth to Cell Tracking Challenge ground truth
        """
        
        t = load_trks(trk_path)
        self.trk_path = trk_path
        lin = t['lineages']
        mask = t['y']
        txt = lineage_dic2txt(lin)
        # save

        fm = ("%0" + str(self.digit_num) + "d") % (self.dt_id)
        os.path.join(self.root, fm+'_GT')
        txt.to_csv(os.path.join(fm, 'TRA', 'man_track.txt'), index=0, sep=' ', header=False)
        save_seq(mask, os.path.join(fm, 'SEG'), 'man_seg', dig_num=self.digit_num, base=self.t_base)
        save_seq(mask, os.path.join(fm, 'TRA'), 'man_track', dig_num=self.digit_num, base=self.t_base)
        return

    def init_ctc_dir(self):
        """Initialize Cell Tracking Challenge directory
        """
        root = self.root
        fm = ("%0" + str(self.digit_num) + "d") % (self.dt_id)
        os.mkdir(os.path.join(root, fm))
        os.mkdir(os.path.join(root, fm+'_RES'))
        os.mkdir(os.path.join(root, fm+'_GT'))
        return

    def evaluate(self):
        os.system(os.path.join(self.path_ctc_software, 'SEGMeasure ') + self.root + ' ' + str(self.dt_id) + ' ' + str(self.digit_num))
        os.system(os.path.join(self.path_ctc_software, 'TRKMeasure ') + self.root + ' ' + str(self.dt_id) + ' ' + str(self.digit_num))
        return