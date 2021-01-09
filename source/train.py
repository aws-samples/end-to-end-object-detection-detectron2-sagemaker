import argparse
import sys
import logging
import os
import torch
import json
import shutil
from torch.nn.parallel import DistributedDataParallel

# import some common detectron2 utilities
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser,\
    default_setup, hooks, launch

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer


from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper
from detectron2.evaluation import COCOEvaluator
import time
import datetime
import numpy as np

# from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

def _register_dataset(dataset_name, annotation_file, image_dir):
    from detectron2.data.datasets import register_coco_instances

    dataset_location = os.environ["SM_CHANNEL_TRAINING"]

    register_coco_instances(dataset_name, {}, os.path.join(dataset_location, annotation_file), 
                            os.path.join(dataset_location, image_dir))

    drone_meta = MetadataCatalog.get(dataset_name)
    logger.info(f"Registered dataset {dataset_name}")
    logger.info(drone_meta)


def _setup(sm_args):
    """
    Create D2 configs and perform basic setups.  
    """

    # Choose whether to use config file from D2 model zoo or 
    # user supplied config file ("local_config_file")
    if sm_args.local_config_file is not None:
        config_file_path = f"/opt/ml/code/{sm_args.local_config_file}"
        config_file = sm_args.local_config_file
    else:
        config_file_path = f"/opt/ml/code/detectron2/configs/{sm_args.config_file}"
        config_file = sm_args.config_file

    # Register custom dataset
    dataset_train_name = "cb_train"
    _register_dataset(dataset_train_name, "train.json", "train")
    
    dataset_val_name = "cb_val"
    _register_dataset(dataset_val_name, "val.json", "val")

    # Build config file
    cfg = get_cfg() # retrieve baseline config: https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
    cfg.merge_from_file(config_file_path) # merge defaults with provided config file
    list_opts = _opts_to_list(sm_args.opts)
    cfg.merge_from_list(list_opts) # override parameters with user defined opts
    cfg.DATASETS.TRAIN = (dataset_train_name,) # define dataset used for training
    cfg.DATASETS.TEST = (dataset_val_name,)  # no test dataset available
    cfg.OUTPUT_DIR = os.environ['SM_OUTPUT_DATA_DIR']
    cfg.TEST.EVAL_PERIOD = 500
    cfg.freeze()
    
    # D2 expects ArgParser.NameSpace object to ammend Cfg node.
    d2_args = _custom_argument_parser(config_file_path, sm_args.opts, sm_args.resume)
    # Perform training setup before training job starts
    default_setup(cfg, d2_args)
    
    return cfg
    

def _custom_argument_parser(config_file, opts, resume):
    """
    Create a parser with some common arguments for Detectron2 training script.
    Returns:
        argparse.NameSpace:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument("--config-file", default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--opts",default=None ,help="Modify config options using the command-line")
    parser.add_argument("--resume", type=str, default="True", help="whether to attempt to resume from the checkpoint directory",)
    
    args = parser.parse_args(["--config-file", config_file,
                             "--resume", resume,
                             "--opts", opts])
    return args


def _opts_to_list(opts):
    """
    This function takes a string and converts it to list of string params (YACS expected format). 
    E.g.:
        ['SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.9999'] -> ['SOLVER.IMS_PER_BATCH', '2', 'SOLVER.BASE_LR', '0.9999']
    """
    import re
    
    if opts is not None:
        list_opts = re.split('\s+', opts)
        return list_opts
    return ""

def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker
    world["is_master"] = current_host == sorted(hosts)[0]

    return world

def _save_model():
    """
    This method copies model weight, config, and checkpoint(optionally)
    from output directory to model directory.
    Sagemaker then automatically archives content of model directory
    and adds it to model registry once training job is completed.
    """
    
    logger.info("Saving the model into model dir")
    
    model_dir = os.environ['SM_MODEL_DIR']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    
    # copy model_final.pth to model dir
    model_path = os.path.join(output_dir, "model_final.pth")
    new_model_path = os.path.join(model_dir, 'model.pth')
    shutil.copyfile(model_path, new_model_path)
    
    
    shutil.copytree('/opt/ml/code/', os.path.join(model_dir, 'code'))

    # copy config.yaml to model dir
    config_path = os.path.join(output_dir, "config.yaml")
    new_config_path = os.path.join(model_dir, "config.yaml")
    shutil.copyfile(config_path, new_config_path)

    try:
        # copy checkpoint file to model dir
        checkpoint_path = os.path.join(output_dir, "last_checkpoint")
        new_checkpoint_path = os.path.join(model_dir, "last_checkpoint")
        shutil.copyfile(checkpoint_path, new_checkpoint_path)
    except Exception:
        logger.debug("D2 checkpoint file is not available.")


class LossEvalHook(HookBase):
    """
    Subclass to extend HookBase functionality to log validation loss every TEST.EVAL_PERIOD 
    """
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        

# def custom_mapper(dataset_dict):
#     """
#     Custom mapper for image augmentation in the CocoTrainer class build_train_loader method 
#     """
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     transform_list = [T.RandomBrightness(0.8, 1.6),
#                       T.RandomCrop('relative_range', (0.9, 0.9))
# #                       T.Resize(1200,1200),
#                       ]
    
#     image, transforms = T.apply_transform_gens(transform_list, image)
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict
        

class CocoTrainer(DefaultTrainer): 
    """
    Subclass that inherits DefaultTrainer to include custom COCOEvaluator and train_loader for image augmentation 
    """
    
    @classmethod 
    def build_evaluator(cls, cfg, dataset_name, output_folder=None): 
        """
        Custom COCOEvaluator for class level mAP evaluation every `cfg.TEST.EVAL_PERIOD` iterations. 
        """
        if not output_folder:
            os.makedirs(os.path.join(os.environ['SM_OUTPUT_DATA_DIR'], "coco_eval"), exist_ok=True)
            output_folder = os.path.join(os.environ['SM_OUTPUT_DATA_DIR'], "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder) # True is distributed 
    
#     @classmethod
#     def build_train_loader(cls, cfg):
#         """
#         Build image augmentation techniques here hoping to augment partially off-screen objects 
#         """
#         return build_detection_train_loader(cfg, mapper=custom_mapper)
    
#     @classmethod
#     def build_test_loader(cls, cfg):
#         """
#         TODO: Do we want to augment the test split too? 
#         """
#         pass
    
    def build_hooks(self):
        """
        Extra hook to collect and plot evaluation metrics into TensorBoard. 
        """
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    
        
def main(sm_args, world):
    
    cfg = _setup(sm_args)
    
    is_zero_rank = comm.get_local_rank()==0
    
    trainer = CocoTrainer(cfg)
    resume = True if sm_args.resume == "True" else False
    trainer.resume_or_load(resume=resume)
    trainer.train()
    
    if world["is_master"] and is_zero_rank:
        _save_model()
    


if __name__ == "__main__":
    
    os.environ['FVCORE_CACHE'] = '/tmp' # Fix for detectron
    # Sagemaker configuration
    logger.info('Starting training...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default="True") # TODO: is it relevant?
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config-file', type=str, default=None, metavar="FILE", help="If config file specificed, then one of the Detectron2 configs will be used. \
                       Refer to https://github.com/facebookresearch/detectron2/tree/master/configs")
    group.add_argument('--local-config-file', type=str, default=None, metavar="FILE", help="If local config file specified, then config file \
                       from container_training directory will be used.")
    parser.add_argument('--opts', default=None)
    sm_args = parser.parse_args()
    
    # Derive parameters of distributed training
    world = get_training_world()
    logger.info(f'Running "nccl" backend on {world["number_of_machines"]} machine(s), \
                each with {world["number_of_processes"]} GPU device(s). World size is \
                {world["size"]}. Current machine rank is {world["machine_rank"]}.')
        
    # Launch D2 distributed training
    launch(
        main,
        num_gpus_per_machine=world["number_of_processes"],
        num_machines=world["number_of_machines"],
        machine_rank=world["machine_rank"],
        dist_url=f"tcp://{world['master_addr']}:{world['master_port']}",
        args=(sm_args, world,),
    )