import pandas
import cv2 as cv
import os

# Inputs: path = dir with parent images with objects in them; export_path = dir for object crops
path = "D:/WHCR_2025/11_FINAL_2025_CLASSIFICATIONS"
export_path ="D:/R2_models/R2_cropsx/"

# Input csv file with unique_image_jpg, xmin, ymin, w, and h of annotation; check on data types
csv_data = pandas.read_csv("D:/R2_models/R2_family_test3.csv")

#csv_data.columns = (['family', 'taxona', 'taxonb', 'unique_image_jpg', 'xmin', 'ymin', 'w', 'h', 'unique_BB'])

#datatypes = csv_data.dtypes
print(csv_data)

dirs = os.listdir(path)
file_list = []
for file in dirs:
    basename = os.path.splitext(file)[0] + ".jpg" # take basename (not path) and add .jpg
    file_list.append(basename)
matches = csv_data[csv_data['unique_image_jpg'].isin(file_list)]
print("Matches: ", len(matches))

for index, row in matches.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = path + row['unique_image_jpg']  # +'.jpg'
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
    cv.imwrite(export_path + row['unique_BB'], crops, [int(cv.IMWRITE_JPEG_QUALITY), 95])




