# This is default implementation of inference_handler: 
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py
# SM specs: https://sagemaker.readthedocs.io/en/stable/using_pytorch.html

import os
import io
import argparse
import logging
import sys
import pickle    
from yacs.config import CfgNode as CN
import numpy as np
from PIL import Image
import cv2
import json

import torch
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import detectron2.data.transforms as T
from detectron2.config import get_cfg


from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
# from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_NPY # TODO: for local debug only. Remove or comment when deploying remotely.
# from six import StringIO, BytesIO  # TODO: for local debug only. Remove or comment when deploying remotely.
import d2_deserializer
import pycocotools.mask as mask_util
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_predictor(config_path, model_path):
    
    cfg = get_cfg()
    
    cfg.merge_from_file(config_path) # get baseline parameters from YAML config
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.20  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_path
    cfg.DATASETS.TEST = ("cb_val", )

    pred = DefaultPredictor(cfg)
    logger.info(cfg)
    eval_results = pred.model.eval()

    return pred


def model_fn(model_dir):
    """
    Deserialize and load D2 model. This method is called automatically by Sagemaker.
    model_dir is location where your trained model will be downloaded.
    """
    
    try:
        config_path = os.path.join(MODEL_EXTRACT_LOC, 'config.yaml')
        
        for file in os.listdir(model_dir):
            logger.info(file)
            if file.endswith(".pth") or file.endswith(".pkl"):
                model_path = os.path.join(model_dir, file)

        logger.info(f"Using config file {config_path}")
        logger.info(f"Using model weights from {model_path}")            

        pred = _get_predictor(config_path, model_path)
        
    except Exception as e:
        logger.error("Model deserialization failed...")
        logger.error(e)  
        
    logger.info("Deserialization completed ...")
    
    return pred


def input_fn(request_body, request_content_type):
    """
    Converts image from NPY format to numpy.
    """
    logger.info(f"Handling inputs...Content type is {request_content_type}")
    
    try:
        if "application/x-npy" in request_content_type:
            input_object = decoder.decode(request_body, CONTENT_TYPE_NPY)
        elif "jpeg" in request_content_type:
            nparr = np.frombuffer(request_body, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            input_object = np.asarray(img)
        else:
            raise Exception(f"Unsupported request content type {request_content_type}")
    except Exception as e:
        logger.error("Input deserialization failed...")
        logger.error(e)  
        return None
            
    logger.info("Input deserialization completed...")
    logger.info(f"Input object type is {type(input_object)} and shape {input_object.shape}")

    return input_object


def predict_fn(input_object, model):
    # according to D2 rquirements: https://detectron2.readthedocs.io/tutorials/models.html
    
    logger.info("Doing predictions...")
    logger.debug(f"Input object type is {type(input_object)} and shape {input_object.shape}")
    logger.debug(f"Predictor type is {type(model)}")
    
    try:
        prediction = model(input_object)
    except Exception as e:
        logger.error("Prediction failed...")
        logger.error(e)
        return None

    logger.debug("Predictions are:")
    logger.debug(prediction)
    
    return prediction


def output_fn(prediction, response_content_type):
    
    logger.info("Processing output predictions...")
    logger.debug(f"Output object type is {type(prediction)}")
        
    try:
        logger.info(f"response_content_type: {response_content_type}")
        if "json" in response_content_type:
            logger.debug('JSON processing -- d2_deserializer')
            output = d2_deserializer.d2_to_json(prediction)

        elif "detectron2" in response_content_type:
            logger.debug("check prediction before pickling")
            logger.debug(type(prediction))
            
            instances = prediction['instances']
            rle_masks = d2_deserializer.convert_masks_to_rle(instances.get_fields()["pred_masks"])
            instances.set("pred_masks_rle", rle_masks)
            instances.remove('pred_masks')
            
            pickled_outputs = pickle.dumps(prediction)
            stream = io.BytesIO(pickled_outputs)
            output = stream.getvalue()
            
        else:
            raise Exception(f"Unsupported response content type {response_content_type}")
        
    except Exception as e:
        logger.error("Output processing failed...")
        logger.error(e)
        return None
    
    logger.info("Output processing completed")
    logger.debug(f"Predicted output type is {type(output)}")

    return output
