# End to End Object Detection on AWS SageMaker with Detectron2 

Author: Calvin Wang (AWS) 

This repository provides an end to end walkthrough on taking a labeled dataset(s) from SageMaker GroundTruth and training and ultimately deploying a Detectron2 model for Object Detection. Furthermore, I've also provided docs on how to [deploy a trained model via AWS CDK](./endpoint_deploy_cdk.md) as well.  

## Requirements 
```bash 
$ aws --version
aws-cli/1.18.137 Python/3.6.7 Linux/4.14.193-113.317.amzn1.x86_64 botocore/1.17.60
``` 

## Contents 
```
./d2_on_sagemaker/
├── README.md
├── requirements.txt
├── d2_eval.ipynb
├── d2_inference.ipynb
├── d2_local.ipynb
├── d2_train.ipynb
├── dataprep
│   ├── SMGT_to_COCO.py
│   ├── dataprep.ipynb
│   └── go_smgt.ipynb
├── source
│   ├── Base-RCNN-FPN.yaml
│   ├── d2_deserializer.py
│   ├── faster_rcnn_R_101_FPN_3x.yaml
│   ├── inference.py
│   ├── requirements.txt
│   └── train.py
└── utils
    ├── build_d2_amzn_linux_1.sh
    ├── build_d2_amzn_linux_2.sh
    └── train_val_loss.py
``` 

- `source/` 
    - `train.py`: handler for Detectron2 in the SageMaker Training Job to navigate the entrypoint and specify hyperparameters passed in at training call. (Refer to [d2_train.ipynb](d2_train.ipynb))
    - `inference.py`: handler for Detectron2 in SageMaker Endpoint to load model and call prediction on an input image. (Refer to [d2_inference.ipynb](d2_inference.ipynb))
    - `d2_deserializer.py`: utility functions for serializing/deserializing between Detectron2 and JSON for lighter API passing 
    - `requirements.txt`: Dependencies for Detectron2 to run in a SageMaker environment 
    - `Base-RCNN-FPN.yaml` and `faster_rcnn_r_101_FPN_3x.yaml`: default config yamls from Detectron2's configuration. See more [here](https://detectron2.readthedocs.io/tutorials/configs.html).
- `dataprep/`
    - `dataprep.ipynb`: takes SageMaker GroundTruth Manifest file and splits into train/val/test splits with multiclass stratification -- ensuring equal representation of each class in each split. 
    - `smgt_coco.py`: helper code to translate dataset from SageMaker GroundTruth Manifest output to COCO format
    - `go_smgt.ipynb`: helper code to translate Google Open dataset into SageMaker GroundTruth format. 
- `d2_local.ipynb`: A notebook for training and predicting a Detectron2 model locally. This will be using a local GPU. 
- `d2_train.ipynb`: A notebook for training a Detectron2 model in a SageMaker Training Job.
- `d2_eval.ipynb`: A notebook for evaluating a Detectron2 model for object detection performance metrics. 
- `d2_inference.ipynb`: A notebook for deploying a Detectron2 model to a SageMaker Endpoint and calling inference on an image through an API call. 

## E2E Training to Deploying Workflow 

0. Install dependencies via `requirements.txt` 
1. Start and complete labeling jobs with SageMaker GroundTruth and have those `output.manifest` S3 keys handy 
2. Run through `dataprep/dataprep.ipynb` 
3. Run `dataprep/SMGT_to_COCO.py` 
4. Upload the output from Step 3 to your desired location in S3. We recommend using `aws s3 sync` from the CLI to somewhere like s3://my-bucket/training-data/.
5. (Optional) Run `d2_local.ipynb` for local training fun. This is great for small experimentations. 
6. Run `d2_train.ipynb` to launch a SageMaker Training Job with Detectron2 
7. Run `d2_eval.ipynb` to evaluate performance metrics such as: 
    - precision / recall
    - average precision
    - mean average precision 
    - class distributions 
    - false detections and true detections 
8. Run `d2_inference.ipynb` to deploy a SageMaker Endpoint with Detectron2 

## Bring Your Own `model.pth` Workflow to Deploy
1. Have your `model.pth` and `config.yaml` available locally 
2. Create a `code` directory and copy and paste all files in from the `source` directory. Your directory should now look like this: 
``` 
.
├── code
│   ├── d2_deserializer.py
│   ├── inference.py
│   ├── requirements.txt
│   └── train.py
├── config.yaml
└── model.pth
```
3. Compress the directory into a `tar.gz` file and upload it to s3 
```
$ tar -zcvf model.tar.gz *
$ aws s3 cp model.tar.gz s3://__PATH__TO__WHERE__YOU__WANT__TO__UPLOAD__TO/
```
4. Run `d2_inference.ipynb` to deploy a SageMaker Endpoint with the `model_url` set to the path you uploaded to in step 3^. 

## Deployment and Inference Benchmarks 
| Instance Type  | Cost per hour ($) | Deployment Time (sec) | Avg time per image (sec) | Avg cost per 1000 images ($) |
|----------------|-------------------|-----------------------|--------------------------|------------------------------|
| ml.p2.xlarge   | $1.125            | 672                   | 0.607                    | $0.19                        |
| ml.g4dn.xlarge | $0.736            | 622                   | 0.217                    | $0.04                        |
| ml.p3.2xlarge  | $3.825            | 623                   | 0.133                    | $0.12                        |
