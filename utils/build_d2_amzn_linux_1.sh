#!/usr/bin/env bash

# Builds Detectron2 on SageMaker (Amazon Linux 1)

source activate pytorch_p36
sudo yum install gcc72 gcc72-c++
sudo update-alternatives --config gcc
export FORCE_CUDA="1"
export TORCH_CUDA_ARCH_LIST="Volta"
conda install pytorch torchvision -c pytorch
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 'git+https://github.com/facebookresearch/fvcore' 'git+https://github.com/facebookresearch/detectron2.git' google-colab scikit-image sagemaker-inference
