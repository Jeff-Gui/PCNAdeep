# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
import torch
import numpy as np
import skimage.measure as measure
import copy
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


def pred2json(masks, labels, fp):
    """Transform detectron2 prediction to via2 json format
    Args:
        masks: list of instance mask in the frame, each mask should contain only one object.
        labels: list of instance label in the frame, should be corresponsing to the mask in order.
        fp: file name for this frame
    
    Return:
        json format readable by VIA2 annotator
    """
    cc_stage = {0:'G1/G2', 1:'S', 2:'M', 3:"E"}
    region_tmp = {"shape_attributes":{"name":"polygon","all_points_x":[],"all_points_y":[]}, "region_attributes":{"phase":''}}

    if len(masks)<1:
        return {}

    tmp = {"filename":fp,"size":masks[0].astype('bool').size,"regions":[],"file_attributes":{}}
    for i in range(len(masks)):
        region = measure.regionprops(measure.label(masks[i], connectivity=1))[0]
        if region.image.shape[0]<2 or region.image.shape[1]<2:
            continue
        # register regions
        cur_tmp = copy.deepcopy(region_tmp)
        bbox = list(region.bbox)
        bbox[0],bbox[1] = bbox[1], bbox[0] # swap x and y
        bbox[2],bbox[3] = bbox[3], bbox[2]
        ct = measure.find_contours(region.image, 0.5)
        if len(ct)<1:
            continue
        ct = ct[0]
        if ct[0][0] != ct[-1][0] or ct[0][1] != ct[-1][1]:
            # non connected
            ct_image = np.zeros((bbox[3]-bbox[1]+2, bbox[2]-bbox[0]+2))
            ct_image[1:-1,1:-1] = region.image.copy()
            ct = measure.find_contours(ct_image, 0.5)[0]
            # edge = measure.approximate_polygon(ct, tolerance=0.001)
            edge = ct
            for k in range(len(edge)): # swap x and y
                x = edge[k][0] - 1
                if x<0: 
                    x=0
                elif x>region.image.shape[0]-1:
                    x = region.image.shape[0]-1
                y = edge[k][1] - 1
                if y<0:
                    y=0
                elif y> region.image.shape[1]-1:
                    y = region.image.shape[1]-1
                edge[k] = [y,x]
            edge = edge.tolist()
            elements = list(map(lambda x:tuple(x), edge))
            edge = list(set(elements))
            edge.sort(key=elements.index)
            edge = np.array(edge)
            edge[:,0] += bbox[0]
            edge[:,1] += bbox[1]
            edge = list(edge.ravel())
            edge += edge[0:2]
        else:
            # edge = measure.approximate_polygon(ct, tolerance=0.4)
            edge = ct
            for k in range(len(edge)): # swap x and y
                edge[k] = [edge[k][1], edge[k][0]]   
            edge[:,0] += bbox[0]
            edge[:,1] += bbox[1]
            edge = list(edge.ravel())
        cur_tmp['shape_attributes']['all_points_x'] = edge[::2]
        cur_tmp['shape_attributes']['all_points_y'] = edge[1::2]
        cur_tmp['region_attributes']['phase'] = cc_stage[int(labels[i])]
        tmp['regions'].append(cur_tmp)
        
    return tmp

def predictFrame(img, frame_id, demonstrator, is_gray=False, size_flt=1000):
    """Predict single frame and deduce meta information
    
    Args:
        img: uint8 image slice, ndarray
        frame_id: index of the slice, int from 0
        demostrator
        size_flt: size filter, int
        is_gray: whether the slice is gray. If true, will convert to 3 channels at first

    """
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
    # 0: G1/G2, 1: S, 2: M, 3: E-early G1
    cls = predictions['instances'].pred_classes
    conf = predictions['instances'].scores
    factor = {0:'G1/G2', 1:'S', 2:'M', 3:'G1/G2'}
    for s in range(mask.shape[0]):
        if np.sum(mask[s,:,:]) < 1000:
            continue
        sc = conf[s].item()
        ori = np.max(mask_slice[mask[s,:,:]!=0])
        if ori!=0:
            if sc>conf[ori-1].item():
                mask_slice[mask[s,:,:]!=0] = s+1
        else:
            mask_slice[mask[s,:,:]!=0] = s+1
    
    props = measure.regionprops_table(mask_slice, intensity_image=img[:,:,0], properties=('label','bbox','centroid','mean_intensity'))
    props = pd.DataFrame(props)
    props.columns = ['label','bbox-0','bbox-1','bbox-2','bbox-3','Center_of_the_object_0','Center_of_the_object_1','mean_intensity']

    img_relabel = measure.label(mask_slice, connectivity=1)
    props_relabel = measure.regionprops_table(img_relabel, properties=('label','centroid'))
    props_relabel = pd.DataFrame(props_relabel)
    props_relabel.columns = ['continuous_label','Center_of_the_object_0', 'Center_of_the_object_1']

    out_props = pd.merge(props, props_relabel, on=['Center_of_the_object_0','Center_of_the_object_1'])
    out_props['frame'] = frame_id
    phase = []
    G_confid = []
    S_confid = []
    M_confid = []
    for row in range(out_props.shape[0]):
        lb = int(out_props.iloc[row][0])
        p = factor[cls[lb-1].item()]
        confid = conf[lb-1].item()
        phase.append(p)
        if p=='G1/G2':
            G_confid.append(confid)
            S_confid.append((1-confid)/2)
            M_confid.append((1-confid)/2)
        elif p=='S':
            S_confid.append(confid)
            G_confid.append((1-confid)/2)
            M_confid.append((1-confid)/2)
        else:
            M_confid.append(confid)
            G_confid.append((1-confid)/2)
            S_confid.append((1-confid)/2)
        
    out_props['phase'] = phase
    out_props['Probability of G1/G2'] = G_confid
    out_props['Probability of S'] = S_confid
    out_props['Probability of M'] = M_confid
    
    del out_props['label']

    return img_relabel.astype('uint8'), out_props 
