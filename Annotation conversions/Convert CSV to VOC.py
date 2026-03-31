import pandas as pd
import json
from pycocotools.coco import COCO
from pascal_voc_writer import Writer
import os

os.chdir("C:/BP/species/")

# Inputs: csv_file= input of csv with annotation data;
# export_json= name of COCO json to export
# width = width of images (pixels), height = height of images (pixels)
# categories = link the name of classes related to its index
csv_file = 'new_test_annot.csv'
export_json = 'new_test_annot.json'
width = 6464
height = 4848

csv_data = pd.read_csv(csv_file)

categories = [
    {"id": 0, "name": "species"}]

#csv_data.columns = (['id','image_id','unique_image_jpg','xmin', 'ymin', 'w','h','label_id'])
#csv_data['label_id']= csv_data['label_id'].astype(int)

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
csv_data['annid'] = csv_data.index

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

##########
###

def coco2voc(ann_file, output_dir):
    coco = COCO(ann_file)
    # cats = class categories
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']
    for img in coco.imgs:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)
        if len(annIds) > 0:
            img_fname = coco.imgs[img]['file_name']
            print("img_fname:" , img_fname)
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            print("label_fname", label_fname)
            writer = Writer(img_fname, coco.imgs[img]['width'], coco.imgs[img]['height'])
            anns = coco.loadAnns(annIds)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
               # print(bbox)
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, bbox[0], bbox[1], bbox[2], bbox[3])

                basename = os.path.basename(label_fname)
                print("basename:", basename)
                writer.save(output_dir+'/'+ basename)
output_dir = 'C:/BP/species/parent_voc/'
coco2voc(ann_file=export_json, output_dir= output_dir)