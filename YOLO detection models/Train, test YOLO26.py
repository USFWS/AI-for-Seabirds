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

# TRAIN A MODEL
# dataset download directory can be updated in
# 'C:\Users\user\AppData\Roaming\Ultralytics\settings.json'

time_start = time.time()

def main():
    model = YOLO("yolo26n.pt")
    model.info()

    ## change to batch = -1
    # fastest -- cache = 'ram' , cache = 'disk', cache = False (slowest)
    results = model.train(data="C:/BP/seabird_detection/DATASETS_results/seabird_detect.yaml",
                          batch= 2, #-1 to use suggestion
                          task="detect", epochs=100, # 100-300 epochs
                          imgsz=1024, patience=15, # typical patience 20-30
                          device= device, max_det=200,workers= 20,
                          cache= "disk", # can be false
                          optimizer= "SGD", lr0 = 0.1, momentum=0.937,
                          name="YOLO26n",
                          project="C:/BP/seabird_detection/DATASETS_results/yolo26n_july9/",
                          amp= True)
if __name__ == "__main__":
    main()