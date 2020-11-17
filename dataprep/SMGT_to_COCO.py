'''Transform SageMaker GroundTruth Manifest Output to COCO Format: Object Detection

Author: Calvin Wang (AWS)

This notebook will take in the inputs from the first cell below and generate the following structure:
```
data/
    {job_name}-train.json 
    {job_name}-val.json 
    {job_name}-test.json 
    train/
        ... imgs ...
    val/
        ... imgs ...
    test/
        ... imgs ...
```

## INPUTS: 
- `job_name`: str = the labeling job name of your GroundTruth Labeling job

## Note: 
- label class 0 is kept for background in accordance to COCO format 
- we will be leaving the `segmentation` field empty in the COCO annotations 
'''

import json
import ntpath
import boto3
from tqdm import tqdm
import os 
import argparse

split_list = ['train', 'val', 'test']
s3_client = boto3.client('s3')
manifest_dir = 'manifests'

def main(job_name): 
    """
    Main method for converting SageMaker GroundTruth manifest output to COCO format. 
    
    Args: 
    - job_name (str): the name of the labeling job name. This should be the `main_job_name` from when you went through `dataprep.ipynb`
    """

    job_name_meta = job_name + '-metadata'
    classnamesids = {}

    for split in split_list: 
        file_name = f'{manifest_dir}/{job_name}-{split}.manifest'
        print(f'PROCESSING: {file_name}')

        if not os.path.exists(f'data/{split}'):
            os.makedirs(f'data/{split}')

        img_dims = []
        data_objs  = []
        input_files = []
        img_id_dims = {}

        with open(file_name) as out_manifest:
            for line in out_manifest:
                data_obj = json.loads(line)
                data_objs.append(data_obj)
                for k, v in data_obj[job_name_meta].get('class-map', {}).items():
                    classnamesids[int(k) + 1] = v # classnameid must start from 1 for coco
        
        for line in tqdm(data_objs):
            chop = line['source-ref'].split('/')
            bucket = chop[2]
            key = '/'.join(chop[3:])
            download_path = f'data/{split}/{chop[-1]}'
            s3_client.download_file(bucket, key, download_path)
            input_files.append(ntpath.basename(line['source-ref']))
            img_dims.append( (line[job_name]['image_size'][0]['height'],
                        line[job_name]['image_size'][0]['width']) )
        
        images = []
        images_key = { 
                    "coco_url": "",
                    "date_captured": "",
                    "flickr_url": "",
                    "license": 0,
                    "id": 0,
                    "file_name": "",
                    "height": 0,
                    "width": 0
                    }   
        classnamesids_map = [ (classnameid, classname) for classnameid, classname in classnamesids.items() ]
        
        for img_id, input_file in enumerate(input_files):
            images_key["file_name"] = input_file
            images_key["id"] = img_id
            h, w = img_dims[img_id]
            images_key['height'] = h
            images_key['width'] = w
            images.append(images_key.copy())
            img_id_dims[img_id] = (h, w)
        
        categories = []
        category = {
            "id": '',
            "name": '',
            "supercategory": ""
                    }
        
        for idd, classname in classnamesids_map:   
            category['id'] = idd
            category['name'] = classname
            categories.append(category.copy())
        
        licenses =  [
            {
            "name": "",
            "id": 0,
            "url": ""
            }
        ]
        
        info =  {
            "contributor": "",
            "date_created": "2020-01-23",
            "description": "",
            "url": "",
            "version": 3,
            "year": "2020"
        }
        
        boxID = 0
        annotations = []
        
        for image_id, file in enumerate(data_objs):
            img_h, img_w = img_id_dims[image_id]
            for bbox in file[job_name]['annotations']:
                coco_bbox = [bbox['left'], bbox['top'], 
                            bbox['width'], bbox['height']] #l, t, w, h
                try: 
                    assert(bbox['top']+bbox['height'] <= img_h)
                    assert(bbox['left']+bbox['width'] <= img_w)
                except: 
                    print(coco_bbox, (img_h, img_w), file, '\n')
                annot = {
                    'id': boxID,
                    'image_id': image_id,
                    'category_id': bbox['class_id']+1, # class0 = background
                    'segmentation': [],
                    'area': bbox['width'] * bbox['height'],
                    'bbox': coco_bbox,
                    'iscrowd': 0
                }
                boxID+=1
                annotations.append(annot)
        
        COCO_json = {
            "licenses": licenses,
            "info": info,
            "categories": categories,
            "images": images,
            "annotations": annotations  
        }
        
        with open(f'data/{split}.json', 'w') as json_file:
            json.dump(COCO_json, json_file)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Args for SageMaker GroundTruth -> COCO format')

    # Add the arguments
    my_parser.add_argument('--job_name',
                        metavar='job_name',
                        type=str,
                        required=True,
                        help='`main_job_name` from when you went through `dataprep.ipynb`')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    
    # Execute main 
    main(args.job_name)