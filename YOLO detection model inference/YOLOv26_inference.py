import time

import torch
from sahi import AutoDetectionModel
import sahi.predict
import os
import csv
import pandas as pd
import config

# Inputs: root_dir = folder with images;
#         new_csv = detection csv to output
#         model_path = path to YOLOv8 weights file
root_dir = config.SOURCE_IMG
new_csv = config.NEW_CSV
export_dir = config.EXPORT_DIR

visual_path = "E:/Selected Sept 30/viz5/"

model_path = config.MODEL_PATH_DETECT

time_start = time.time()

device = "cuda:0"
print(f"Using {device} device")

detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path,
    confidence_threshold=0.25,
    device=device
)

detection_model.model.overrides['half'] = True
detection_model.model.overrides['imgsz'] = 2048

# Write header, storing bbox/class/score as clean numeric/string columns from the start
with open(new_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['unique_image_jpg', 'xmin', 'ymin', 'xmax', 'ymax', 'w', 'h', 'class', 'score'])

x = 0
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith("jpg"):
            source = os.path.join(root, filename)
            print(source)
            x += 1
            print(x)

            result = sahi.predict.get_sliced_prediction(
                source,
                detection_model,
                slice_height=2048,
                slice_width=2048,
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.0,
               # postprocess_match_metric='IOU',
               # postprocess_type='GREEDYNMM',
                # postprocess_match_threshold=0.20
            )
            object_prediction_list = result.object_prediction_list

            base = os.path.basename(filename)
            result.export_visuals(
                file_name=base,
                export_dir= visual_path,
                hide_labels=False, hide_conf=False, rect_th=3
            )

            # Pull numeric values directly from SAHI objects instead of
            # stringifying and later regex-parsing them back out.
            with open(new_csv, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for pred in object_prediction_list:
                    bbox = pred.bbox  # sahi.prediction.PredictionResult bbox object
                    xmin = bbox.minx
                    ymin = bbox.miny
                    xmax = bbox.maxx
                    ymax = bbox.maxy
                    w = xmax - xmin
                    h = ymax - ymin

                    class_name = pred.category.name
                    score = pred.score.value

                    writer.writerow([
                        base,  # store just the basename, no need to strip later
                        round(xmin), round(ymin), round(xmax), round(ymax),
                        round(w), round(h),
                        class_name,
                        score
                    ])

# Post-process: just build the unique_BB id, everything else is already clean
csv_data = pd.read_csv(new_csv)
print(csv_data)

csv_data['temp_name'] = csv_data['unique_image_jpg'].str.replace(r".jpg", '', regex=True)

csv_data['unique_BB'] = (
    csv_data['temp_name'] + "_" +
    csv_data['xmin'].astype(str) + "_" +
    csv_data['ymin'].astype(str) + "_" +
    csv_data['w'].astype(str) + "_" +
    csv_data['h'].astype(str)
)

del csv_data['temp_name']
del csv_data['xmax']
del csv_data['ymax']

csv_data.to_csv(new_csv, index=False)

time_end = time.time()

duration = time_end - time_start
duration_in_minutes = duration / 60

print("Minutes:", duration_in_minutes)