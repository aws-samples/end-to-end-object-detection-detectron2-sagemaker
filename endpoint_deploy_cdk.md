# Deploying a SageMaker Training Job Output to SageMaker Endpoint with AWS CDK

This walkthrough will demonstrate how to deploy a SageMaker Endpoint from the output of a SageMaker Training Job with AWS CDK. Assuming you've followed the project's directions so far and have a SageMaker Training Job ouptut `model.tar.gz` with the following structure: 

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

You will now want to add these parts into your CDK deployment to be able to deploy your trained model into your stack. Note that the `image` for the `CfnModel` currently reflects a PyTorch 1.6 GPU instance on Ubuntu, so change accordingly as see fit from available [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md): 

```python
from aws_cdk import aws_sagemaker

# Create a SageMaker Model from the model.tar.gz 

model_s3_path = "s3://<_PATH_TO_TRAINING_OUTPUT_>/model.tar.gz"

model = aws_sagemaker.CfnModel(
    self,
    id= <__MODEL_NAME__>,
    execution_role_arn= <__IAM_ROLE_ARN__>,
    primary_container=aws_sagemaker.CfnModel.ContainerDefinitionProperty(
        image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.6.0-gpu-py36-cu101-ubuntu16.04",
        model_data_url=model_s3_path,
    ),
)

endpoint_config = aws_sagemaker.CfnEndpointConfig(
    self,
    id= <__ENDPOINT_CONFIG_NAME__>,
    production_variants=[
        aws_sagemaker.CfnEndpointConfig.ProductionVariantProperty(
            initial_instance_count=1,
            initial_variant_weight=1.0,
            instance_type="ml.p2.xlarge",
            model_name=model.attr_model_name,
            variant_name="AllTraffic",
        )
    ],
)

endpoint_config.add_depends_on(self.model)

endpoint = aws_sagemaker.CfnEndpoint(
    self,
    id= <__ENDPOINT_NAME__>,
    endpoint_config_name=endpoint_config.attr_endpoint_config_name
)
```