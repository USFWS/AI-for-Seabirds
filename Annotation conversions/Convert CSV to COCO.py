import os
import pandas as pd
import json
import csv
import numpy as np
from sympy import factor

import config

os.chdir("D:/2025/detection/")

# Inputs: csv_file= input of csv with annotation data;
# export_json= name of COCO json to export
# width = width of images (pixels), height = height of images (pixels)
# categories = link the name of classes related to its index
csv_file = config.CSV_DATA
export_json = config.JSON_OUTPUT
width = 1024
height = 1024

csv_data = pd.read_csv(csv_file)
print(csv_data)

categories = [{"label_id": 0, "name": "species"}]

#csv_data.columns = (['id','image_id','unique_image_jpg','xmin', 'ymin', 'w','h','label_id'])

csv_data['annid'] = csv_data.index

len1 = len(csv_data)
print(len1)

# create id for each label
id_list1 = []
for i in range(len1):
    id_list1.append(i)
csv_data['id'] = id_list1

# finds all unique images and maps to integer
image_id1 = pd.unique(csv_data['unique_image_jpg'])
csv_data['image_id'], unique_labels = csv_data['unique_image_jpg'].factorize()
csv_data['image_id']= csv_data['image_id'].astype(int)

csv_data['id']= csv_data['id'].astype(int)

# Create lists to fill in, including nested dictionaries
images = []
annotations = []

def image(row):
    image = {}
    image["width"] = width
    image["height"] = height
    image["id"] = row.image_id
    image["file_name"] = row.unique_image_jpg
  # image["observer"] = row.author # if needed
    return image
def annotation(row):
    annotation = {}
    annotation["id"] = row.id
    annotation["image_id"] = row.image_id
    annotation["category_id"] = row.label_id
   # annotation["segmentation"] = []
    annotation["bbox"] = [row.xmin, row.ymin, row.w, row.h]
    annotation["ignore"] = 0
    annotation["iscrowd"] = 0
    annotation["area"] = (row.h * row.w)
    return annotation

# Iterates through rows
for index, row in csv_data.iterrows():
    annotations.append(annotation(row))
    images.append(image(row))
len(images)

# remove duplicate images
images2 = []

imagedf = csv_data.drop_duplicates(subset=['image_id'])
for index, row in imagedf.iterrows():
    images2.append(image(row))
len(images2)

data_coco = {}
data_coco["images"] = images2
data_coco["categories"] = categories
data_coco["annotations"] = annotations

json.dump(data_coco, open(export_json,"w"), indent=0)
print ("Completed!")