# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
import torch
import numpy as np
import skimage.measure as measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import pandas as pd

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        ).set(thing_classes=['G1/G2', 'S', 'M', 'E'])
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, vis=True):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        if vis==False:
            return predictions
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )            
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output



class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

def predictFrame(img, frame_id, demonstrator, is_gray=True):
    if is_gray:
        img = np.stack([img, img, img], axis=2)  # convert gray to 3 channels
    # Generate mask or visualized output
    predictions = demonstrator.run_on_image(img)[0]
    #print(predictions['instances'].pred_classes)
    # Generate mask
    mask = predictions['instances'].pred_masks
    mask = mask.char().cpu().numpy()
    mask_slice = np.zeros((mask.shape[1], mask.shape[2])).astype('uint8')

    # For visualising class prediction
    # 0: I, interphase, 1: M, mitosis
    cls = predictions['instances'].pred_classes
    conf = predictions['instances'].scores
    factor = {0:'I', 1:'M'}
    for s in range(mask.shape[0]):
        sc = conf[s].item()
        ori = np.max(mask_slice[mask[s,:,:]!=0])
        if ori!=0:
            if sc>conf[ori-1].item():
                mask_slice[mask[s,:,:,]!=0] = s+1
        else:
            mask_slice[mask[s,:,:]!=0] = s+1
    
    img_bin = remove_small_objects(mask_slice,1000)
    img_bin = binary_fill_holes(img_bin.astype('bool'))
    props = measure.regionprops_table(measure.label(img_bin), intensity_image=img[:,:,0], properties=('label','bbox','centroid','mean_intensity'))
    props = pd.DataFrame(props)
    props.columns = ['label','bbox-0','bbox-1','bbox-2','bbox-3','Center_of_the_object_0','Center_of_the_object_1','mean_intensity']

    img_relabel = measure.label(img_bin)
    props_relabel = measure.regionprops_table(img_relabel, properties=('label','centroid'))
    props_relabel = pd.DataFrame(props_relabel)
    props_relabel.columns = ['continuous_label','Center_of_the_object_0', 'Center_of_the_object_1']

    out_props = pd.merge(props, props_relabel, on=['Center_of_the_object_0','Center_of_the_object_1'])
    out_props['frame'] = frame_id
    phase = []
    I_confid = []
    M_confid = []
    for row in range(out_props.shape[0]):
        lb = int(out_props.iloc[row][0])
        p = factor[cls[lb-1].item()]
        confid = conf[lb-1].item()
        phase.append(p)
        if p=='I':
            I_confid.append(confid)
            M_confid.append(1-confid)
        elif p=='M':
            I_confid.append(1-confid)
            M_confid.append(confid)
        
    out_props['phase'] = phase
    out_props['Probability of I'] = I_confid
    out_props['Probability of M'] = M_confid
    
    del out_props['label']

    return img_relabel.astype('uint8'), out_props 
