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
torch.backends.cudnn.enabled=True
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# TRAIN A MODEL
# dataset download directory can be updated in 'C:\Users\aware\AppData\Roaming\Ultralytics\settings.json'

model = YOLO("yolo11m.pt")
model.info()

results = model.train(data="D:/species_2025/12_species_detection/7_opt.yaml",
                      batch=12, iou = 0.20,
                      task="detect", epochs=70, imgsz=1024, patience=0,
                      device=2, max_det=20, lr0=0.001, conf=0.20,
                      cache="False",
                      project="D:/species_2025/12_species_detection/f_model_results/",
                      name="YOLO11m_conf20_Aug3", amp= True, workers=0)

# Load model for inference, then
# model.export(format = "onnx")
