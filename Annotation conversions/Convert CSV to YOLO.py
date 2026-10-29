from tqdm import tqdm
import shutil
import os
import pandas as pd
import json
import config

# Inputs: csv_file= input of csv with annotation data;
# export_json= name of COCO json to export
# width = width of images (pixels), height = height of images (pixels)
# categories = link the name of classes related to its index

os.chdir("H:/seabird_detection/DATASETS_results/")

csv_data = config.CSV_DATA
export_json = 'temp9.json'
img_width = 1024
img_height = 1024
output_path = config.OUTPUT_DIR
##

categories = [{"label_id": 0, "name": "species"}]
csv_data = pd.read_csv(csv_data)
# finds all unique images and maps to integer
image_id1 = pd.unique(csv_data['unique_image_jpg'])
csv_data['image_id'], unique_labels = csv_data['unique_image_jpg'].factorize()
csv_data['image_id']= csv_data['image_id'].astype(int)
csv_data['annid'] = csv_data.index
print(csv_data)

# Create lists to fill in, including nested dictionaries
images = []
annotations = []

def image(row):
    image = {}
    image["width"] = img_width
    image["height"] = img_height
    image["id"] = row.image_id
    image["file_name"] = row.unique_image_jpg
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

# remove duplicate images
images2 = []

imagedf = csv_data.drop_duplicates(subset=['image_id'])
for index, row in imagedf.iterrows():
    images2.append(image(row))

data_coco = {}
data_coco["images"] = images2
data_coco["categories"] = categories
data_coco["annotations"] = annotations

json.dump(data_coco, open(export_json,"w"), indent=0)
print ("Completed json!")

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    ----------
    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def convert_coco_json_to_yolo_txt(output_path, json_file):
    path = make_folders(output_path)
    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        img_id = image["id"]
        img_name = image["file_name"]  # supposed to be file_name
        print("img name", img_name)
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        print("anno_txt", anno_txt)

        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("Converting COCO Json to YOLO txt finished!")

# Inputs: (directory for yolo outputs, coco json input file)
convert_coco_json_to_yolo_txt(output_path, export_json)