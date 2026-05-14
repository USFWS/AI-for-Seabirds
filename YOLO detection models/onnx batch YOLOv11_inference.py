import csv
import torch
import os
import time
from sahi import AutoDetectionModel
import sahi
from sahi.predict import get_sliced_prediction
import pickle

# Inputs:
# source_img = folder with images;
# new_csv = detection csv to output
# visual_path = if visuals specified
# model_path = path to YOLOv8 weights file
# from torch import init_num_threads
# project name where results are expected; will end in /exp/ when created

source_img = "C:/BP/demo/jpgs/"
new_csv = "C:/BP/demo/newbee.csv"

visual_path = "C:/BP/demo/viz/"
model_path = "E:/WHCR_2025/whcr_detector_yolo11s_Aug3.onnx"

project_name = "C:/BP/demo/demo_results/"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

time_start = time.time()

#if not os.path.exists(visual_path):
 #   os.mkdir(visual_path)

#x1 = numexpr.detect_number_of_threads()
# print("Found ", x1, "cores!")

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path,
    device="cuda:0",
)

result = sahi.predict.predict(
    detection_model=detection_model,
    model_type='ultralytics',
    model_confidence_threshold=0.15,
    slice_height=1024,
    slice_width=1024,
    no_standard_prediction=True,
    no_sliced_prediction=False,
    overlap_height_ratio=0.0,
    overlap_width_ratio=0.0,
    model_device='cuda:0',
    source=source_img,
    export_crop=False,
    export_pickle=True,
    novisual=True,
    verbose = 0, # 1 or 2
    project = project_name
   # "E:/WHCR_2025/detection/demo_results/"
)
## Duration tracking
time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Completed in, seconds:", duration_sec)
print("Completed in, minutes:", duration_min)
print("Completed in, hours:", duration_hrs)

## Reformat pickles
###########

pickle_dir = project_name + "exp/pickles/"
print(pickle_dir)

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', "xmin", "ymin", "w", "h", "class", "score"])
x=0
for root, dirs, files in os.walk(pickle_dir):
	for filename in files:
		path = os.path.join(root, filename)
		list_file = open(path, 'rb')
		list2 = pickle.load(list_file)
		total = len(list2)
		#print("total", total)
		basename = os.path.basename(path)
		basename = basename.removesuffix('.pickle')
		basename = basename + ".jpg"

		for i in range(0, total):
			score = list2[i].score.value
			cat = list2[i].category.id
			bbox = list2[i].bbox.to_coco_bbox()
			xmin = bbox[0]
			xmin = int(xmin)
			ymin = bbox[1]
			ymin = int(ymin)
			w = bbox[2]
			w = int(w)
			h = bbox[3]
			h = int(h)

			print(basename, bbox, cat, score)
			with open(new_csv, 'a', newline='') as file:
				writer = csv.writer(file)
				for j in range(1, total):
					writer.writerow([basename, xmin, ymin, w, h, cat, score])