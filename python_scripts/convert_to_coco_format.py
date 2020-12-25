import json
import cv2
import os
from sys import platform
linux = False
if(platform == 'linux' or platform == 'linux2'):
    linux = True
train_or_val = 'validation' #should be either validation or train
if train_or_val == 'train':
    annotations_file_name = "annotationsTrainOnly.txt"
    result_json_file_name = 'cocoformatTrain-oneindexed.json'
else:
    annotations_file_name = "annotationsValidationOnly.txt"
    result_json_file_name = 'cocoformatValidation-oneindexed.json'
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
imgs_path = os.path.join(root, train_or_val)
annotations_file_path = os.path.join(root, annotations_file_name)



coco_dict = {"info": {
        "year": "2020",
        "version": "1",
        "description": "our data set converted into a coco dataset format",
        "contributor": "Gidi and Gali",
        "date_created": "2020-12-16T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://",
            "name": "data"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "Green Bus",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "Yellow Bus",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "White Bus",
            "supercategory": "none"
        },
        {
            "id": 4,
            "name": "Grey Bus",
            "supercategory": "none"
        },
        {
            "id": 5,
            "name": "Blue Bus",
            "supercategory": "none"
        },
        {
            "id": 6,
            "name": "Red Bus",
            "supercategory": "none"
        }
    ],
    "images": [],
    "annotations": []
}

""" this is the structure for images dict items
{
            "id": 0,
            "license": 1,
            "file_name": "0001.jpg",
            "height": 275,
            "width": 490,
            "date_captured": "2020-07-20T19:39:26+00:00"
        }
"""
""" this is the structure for annotations dict items
{
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                45,
                2,
                85,
                85
            ],
            "area": 7225,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                324,
                29,
                72,
                81
            ],
            "area": 5832,
            "segmentation": [],
            "iscrowd": 0
        }
"""


anns_id = 0
file = open(annotations_file_path, mode="r")
for im_num, line in enumerate(file):
    print(line)
    img_name = line.split(':')[0] #a string
    annotations_per_image = line.split(':')[1].split('],')
    for i, box in enumerate(annotations_per_image):
        bb_box = box[1:]
        if i==len(annotations_per_image)-1:
            bb_box = bb_box[:-2]
        bb_box = bb_box.split(',') # we now have a list of the form ['100', '200', '300', '400', '5']
        bb_box = [int(item) for item in bb_box] # we now have a list of the form [100, 200, 300, 400, 5]
        print(bb_box)
        coco_dict["annotations"].append({
            "id": anns_id+i,
            "image_id": im_num,
            "category_id": bb_box[-1],
            "bbox": bb_box[0:-1],
            "area": bb_box[2]*bb_box[3],
            "segmentation": [],
            "iscrowd": 0
        })
    if linux:
        img_obj = cv2.imread(imgs_path+'/'+img_name)
    else:
        img_obj = cv2.imread(imgs_path + '\\' + img_name)
    im_height, im_width, channels = img_obj.shape
    coco_dict["images"].append({
            "id": im_num,
            "license": 1,
            "file_name": img_name,
            "height": im_height,
            "width": im_width,
            "date_captured": "2020-12-20T19:39:26+00:00"
        })
    anns_id += len(annotations_per_image)

file.close()
with open(result_json_file_name, 'w') as json_file:
    json.dump(coco_dict, json_file)



