import torch
from sahi import AutoDetectionModel
import sahi.predict
import csv
import pandas as pd
import os
import time
import sahi

# Inputs: root_dir = folder with images;
#         new_csv = detection csv to output
#         model_path = path to YOLOv8 weights file
# from torch import init_num_threads

root_dir = "E:/WHCR_2025/detection/demo/"
file_type = "jpg"

new_csv = "E:/WHCR_2025/detection/newbee.csv"
visual_path = "E:/WHCR_2025/detection/temp_viz"

#new_csv = "D:/species_2025/12_species_detection/8_inference/survey_095400_yolo10x_conf_20.csv"
model_path = "E:/WHCR_2025/whcr_detector_yolo11s_Aug3.onnx"
# model_path = YOLO(model_path)

#model_path= "E:/WHCR_2025/whcr_detector_yolo11s_Aug3.onnx"

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
    confidence_threshold=0.20,
    device="cuda:0",
    # or 'cuda:0'
)

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', "bbox", "class", "score"])
x=0
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(file_type):
            source = os.path.join(root, file)
            print(source)
            x = x + 1
            print(x)
            result = sahi.predict.get_sliced_prediction(
                source,
                detection_model,
                slice_height=1024,
                slice_width=1024,
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.0
            )
            object_prediction_list = result.object_prediction_list
            base = os.path.basename(file)
      #      result.export_visuals(file_name=base, export_dir= visual_path,
       #            hide_labels=False, hide_conf=False, rect_th=3)

            with open(new_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                for result1 in object_prediction_list:
                    writer.writerow([source, result1.bbox, result1.category, result1.score])


time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Completed in, seconds:", duration_sec)
print("Completed in, minutes:", duration_min)
print("Completed in, hours:", duration_hrs)

csv_data = pd.read_csv(new_csv)
print(csv_data)
csv_data['unique_image_jpg'] = csv_data['unique_image_jpg'].apply(os.path.basename)
bbox = csv_data['bbox']
csv_data['score'] = csv_data['score'].str.replace(r"PredictionScore: <value: ", '', regex=True)
csv_data['score'] = csv_data['score'].str.replace(r">", '', regex=True)

csv_data['class'] = csv_data['class'].str.replace(r"Category: <id:", '', regex=True)
csv_data['class'] = csv_data['class'].str.replace(r" 0, name: ", '', regex=True)
csv_data['class'] = csv_data['class'].str.replace(r" 1, name: ", '', regex=True)
csv_data['class'] = csv_data['class'].str.replace(r" 2, name: ", '', regex=True)
csv_data['class'] = csv_data['class'].str.replace(r">", '', regex=True)
print(csv_data['bbox'])
csv_data['bbox'] = csv_data['bbox'].str.replace(r"BoundingBox: <", '', regex= True)
csv_data['bbox'] = csv_data['bbox'].str.replace(r">", '', regex= True)
print("OKAY")


csv_data[['xmin', 'ymin', 'xmax', 'ymax', 'w', 'h']] = csv_data['bbox'].str.split(',', expand=True)
csv_data['h'] = csv_data['h'].str.replace(r"h: ", '', regex= True)
csv_data['w'] = csv_data['w'].str.replace(r"w: ", '', regex= True)
csv_data['xmin'] = csv_data['xmin'].str.replace(r"(", '', regex= False)

csv_data['temp_name'] = csv_data['unique_image_jpg'].str.replace(r".jpg", '', regex= True)

csv_data['xmin'] = csv_data['xmin'].astype(float).round().astype(int).astype(str)
csv_data['ymin'] = csv_data['ymin'].astype(float).round().astype(int).astype(str)
csv_data['w'] = csv_data['w'].astype(float).round().astype(int).astype(str)
csv_data['h'] = csv_data['h'].astype(float).round().astype(int).astype(str)

csv_data['unique_BB'] = csv_data['temp_name'] + "_" + csv_data['xmin'] + "_" + csv_data['ymin'] + "_" + csv_data['w']+ "_" + csv_data['h']
del csv_data['bbox']
del csv_data['xmax']
del csv_data['ymax']
del csv_data['temp_name']

csv_data.to_csv(new_csv)

time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Completed in, seconds:", duration_sec)
print("Completed in, minutes:", duration_min)
print("Completed in, hours:", duration_hrs)

