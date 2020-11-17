import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import json
import torch
import pycocotools.mask as mask_util
import numpy as np
from detectron2.structures import Instances, Boxes


def json_to_d2(pred_dict, device):
    """ 
    Client side helper function to deserialize the JSON msg back to d2 outputs 
    """
    
    # pred_dict = json.loads(predictions)
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(v).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(v).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(v).to(device).to(torch.uint8)
    
    height, width = pred_dict['image_size']
    del pred_dict['image_size']

    inst = Instances((height, width,), **pred_dict)
    
    return {'instances':inst}


def d2_to_json(predictions):
    """
    Server side helper function to serialize the d2 detections into JSON for API passing 
    """
    
    instances = predictions["instances"]
    output = {}

    # Iterate over fields in Instances
    for k,v in instances.get_fields().items():
        
        if k in ["scores", "pred_classes"]:
            output[k] = v.tolist()
            
        if k=="pred_boxes":
            output[k] = v.tensor.tolist()
            
    output['image_size'] = instances.image_size
    output = json.dumps(output)
    
    return output
