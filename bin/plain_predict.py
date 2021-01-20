import os
import cv2
import skimage.io as io
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

def predict(weight, config, image, stack=False):
    """
    Args:
        weight: path to model
        config: path to config file of the model
        image: can be 2D or 2D+T stack in the order CHWT
        stack: flag of whether the image is a stack or not. If stack used, time should be the first dimension
    
    Return:
        model prediction
    """

    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weight
    predictor = DefaultPredictor(cfg)

    if stack:
        outputs = []
        for i in range(image.shape[0]):
            im = image[i,:,:]
            outputs.append(predictor(im))
    else:
        outputs = [predictor(im)]
    # Visualization
    '''
    v = Visualizer(im[:, :, ::-1],
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Result',v.get_image()[:, :, ::-1])
    cv2.waitKey()
    '''
    return outputs

if __name__ == "__main__":
    root = '/home/zje/detectron2/projects/PCNAdeep/output'
    image_root = '/home/zje/dataset/10A_0103_s26.tif'
    image = io.imread(image_root).astype('float16')
    p = predict(weight=os.path.join(root, 'model_final.pth'),
            config=os.path.join(root, 'config.yaml'),
            image=image,
            stack=True)
    print(p[0])    
