
from ultralytics import YOLO
import openvino
import onnxruntime as ort
import tensorrt

#class UltralyticsDetectionModel(DetectionModel):
  #  """Detection model for Ultralytics YOLO models.
   # Supports PyTorch (.pt), ONNX (.onnx), OpenVINO (.xml or _openvino_model/),
    #NCNN (.param or _ncnn_model/), and TorchScript (.torchscript) models.
    #"""

model_path = "E:/WHCR_2025/whcr_detector_yolo11s_Aug3.pt"
# Load the latest YOLO26 model
model = YOLO(model_path)

# Export the model to ## format with half-precision enabled
# The 'half=True' argument converts weights to FP16
model.export(format= "onnx", half=True)
model.export(format= "engine", half=True)
model.export(format= "openvino", half=True)



