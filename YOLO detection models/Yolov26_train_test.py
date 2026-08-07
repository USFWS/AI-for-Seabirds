# We are grateful for Ultralytics' work in this area of detection research! It is fantastic!
# Please see the tutorial here for more guidance on how to start: https://docs.ultralytics.com/quickstart/ and
# https://docs.ultralytics.com/models/yolov8/
# Also, see here for the Github repository for yolov5: https://github.com/ultralytics/ultralytics

# All annotation data must be in YOLO format
# Your imagery and labels must be in a specific folder structure: /directory1/train/images and directory/train/labels AND
# /directory1/val/images and /directory1/val/labels ; These file paths will be specified in your opt.yaml file; please see template
# in this repository

# Available YOLOv8 models include yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Once the Python requirements are met, you can specify vairables such as batch size, iou, epochs, imgsz (image size),
# patience, device, max_det (maximum detections), project and name where to save the results

from ultralytics import YOLO
import torch
import time

torch.backends.cudnn.enabled=True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Input yaml file
yaml_path = "D:/seabird_detection/DATASETS_results/seabird_detect.yaml"
# TRAIN A MODEL YOLOv26 model
# YOLO options:
# Load a model
model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights
model.info()

# Train the model
results = model.train(data= yaml_path,
                      batch=4, optimizer = "SGD",
                      epochs = 120,
                      task="detect", imgsz=1024, patience=15,
                      device=device, max_det=100, lr0=0.1,
                      cache="disk", workers = 18,
                      project="C:/BP/seabird_yolo_s_/",
                      name="yolov26s_July8")