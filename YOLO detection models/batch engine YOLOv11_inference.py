import csv
import torch
import os
import time
from sahi import AutoDetectionModel
import sahi
from sahi.predict import get_sliced_prediction
import pickle
import tensorrt
from ultralytics import YOLO
import config

# Inputs:
# source_img = folder with images;
# new_csv = detection csv to output
# visual_path = if visuals specified
# model_path = path to YOLOv8 weights file
# from torch import init_num_threads
# project name where results are expected; will end in /exp/ when created
source_img = config.SOURCE_IMG
new_csv = config.NEW_CSV
# visual_path = config.VISUAL_PATH
model_path = config.MODEL_PATH
project_name = config.PROJECT_NAME

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# allow cuDNN to find the fastest conv algorithms for your hardware
torch.backends.cudnn.benchmark = True

time_start = time.time()
yolo_model = YOLO(model_path, task= 'detect')

#if not os.path.exists(visual_path):
 #   os.mkdir(visual_path)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
	model = yolo_model, # pass object, not the path
    model_path=model_path,
    device="cuda:0",
)

 # GPU warmup ────────────────────────────────────────────────────────────────
# TensorRT engines are slow on the first few forward passes while CUDA
# allocates memory and warms up kernels. Warming up before timing ensures
# the clock starts only when the GPU is at full speed.
print("Warming up GPU...")
dummy = torch.zeros(1, 3, 1024, 1024, dtype=torch.float16, device=device)
for _ in range(3):
    detection_model.model(dummy)
torch.cuda.synchronize()
print("Warmup done.")
# detection_model.model.half()

result = sahi.predict.predict(
    detection_model=detection_model,
    model_type='ultralytics',
    model_confidence_threshold=0.15,
    slice_height=1024,
    slice_width=1024,
    no_standard_prediction=True,
    no_sliced_prediction=False,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    model_device= device,
    source=source_img,
    export_crop=False,
    export_pickle=True,
    novisual=True,
    verbose = 0, # 1 or 2
    project = project_name
   # "E:/WHCR_2025/detection/demo_results/"
)
## Duration tracking
torch.cuda.synchronize()
time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Completed in, seconds:", duration_sec)
print("Completed in, minutes:", duration_min)
print("Completed in, hours:", duration_hrs)

## Reformat pickles
pickle_dir = project_name + "exp/pickles/"
print("Reading pickles from: ", pickle_dir)

rows = []
x=0

for root, dirs, files in os.walk(pickle_dir):
	for filename in files:
		path = os.path.join(root, filename)

		with open(path, 'rb') as file:
			detections = pickle.load(file)

		basename = os.path.splitext(filename)[0] + ".jpg"

		for det in detections:
			score = det.score.value
			cat = det.category.id
			x, y, w, h = [int(v) for v in det.bbox.to_coco_bbox()]
			rows.append([basename, x, y, w, h, cat, score])

with open(new_csv, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['unique_image_jpg', 'xmin', 'ymin', 'w', 'h', 'class', 'score'])
	writer.writerows(rows)  # writerows() is much faster than repeated writerow()

	print(f"Wrote {len(rows)} detections to {new_csv}")