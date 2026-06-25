import pandas
import cv2 as cv
import os

import config

# Inputs: path = dir with parent images with objects in them; export_path = dir for object crops
source_img = config.SOURCE_IMG
export_dir = config.EXPORT_DIR

# Input csv file with unique_image_jpg, xmin, ymin, w, and h of annotation; check on data types
csv_data = pandas.read_csv(config.CSV_DATA)

#csv_data.columns = (['family', 'taxona', 'taxonb', 'unique_image_jpg', 'xmin', 'ymin', 'w', 'h', 'unique_BB'])

#datatypes = csv_data.dtypes
print(csv_data)

dirs = os.listdir(source_img)
file_list = []
for file in dirs:
    basename = os.path.splitext(file)[0] + ".jpg" # take basename (not path) and add .jpg
    file_list.append(basename)
matches = csv_data[csv_data['unique_image_jpg'].isin(file_list)]
print("Matches: ", len(matches))

for index, row in matches.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = source_img + row['unique_image_jpg']  # +'.jpg'
    print("Source: ", source_path)
    temp1 = cv.imread(source_path, cv.IMREAD_COLOR)  # this is good

    temp1.shape
    x = row['xmin'] - 10
    if x < 0:
        x = 0
    y = row['ymin'] - 10
    if y < 0:
        y = 0

    w = row['w'] + 20
    h = row['h'] + 20
   # print(x, y, w, h)

    crops = temp1[y:(y + h), x:(x + w)]
    cv.imwrite(export_dir + row['unique_BB'], crops, [int(cv.IMWRITE_JPEG_QUALITY), 95])