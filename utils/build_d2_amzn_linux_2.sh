#!/usr/bin/env bash

# Builds Detectron2 on SageMaker (Amazon Linux 2)

export FORCE_CUDA="1"
export TORCH_CUDA_ARCH_LIST="Volta"

pip install torchvision torch --upgrade 
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 'git+https://github.com/facebookresearch/fvcore' 'git+https://github.com/facebookresearch/detectron2.git' google-colab scikit-image sagemaker-inference
