
import csv
import pandas as pd
import os

new_csv = "D:/seabird_winter/YOLOv11_winter_seabirds_121k_draft.csv"

print("OKAY")
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

csv_data['bbox'] = csv_data['bbox'].str.replace(r"BoundingBox: <", '', regex= True)
csv_data['bbox'] = csv_data['bbox'].str.replace(r">", '', regex= True)

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