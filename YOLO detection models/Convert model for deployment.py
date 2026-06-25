
from ultralytics import YOLO
import openvino
import onnxruntime as ort
import tensorrt

import config

model_path = config.MODEL_PATH
model = YOLO(model_path)

# Export the model to ## format with half-precision enabled
# The 'half=True' argument converts weights to FP16
model.export(format= "onnx", half=True)
model.export(format= "engine", half=True, device= "cuda:0", imgsz=1024, workspace=6,
simplify=True, batch=1)
model.export(format= "openvino", half=True)