
import pandas
import cv2 as cv
import os
from PIL import Image, ImageDraw, ImageFont

##Input: path = dir with parent images; csv_data = detection csv from inference
# export_path_[bird/nonbird/artif] = specify folders for each class
source_img = "C:/Brad/a_detection_of_seabirds/tiles_label2_birds/"
annot_data = pandas.read_csv("C:/Brad/a_detection_of_seabirds/annot not reviewed2.csv")
export_path_bird = "C:/Brad/a_detection_of_seabirds/tiles_w_boxes/"

#export_path_nonbird = "D:/detection_crops_for_inference_2024/2024_Jan5_nonbird_crops/"
#export_path_artif = "D:/detection_crops_for_inference_2024/2024_Jan5_artif_crops/"
if not os.path.exists(export_path_bird):
    os.mkdir(export_path_bird)

annot_data.columns = (['image_id', 'x_min', 'y_min', 'w', 'h', 'label_id', 'unique_image_jpg', 'unique_BB'])
#print(annot_data)
annot_data['unique_image_jpg'] = annot_data['unique_image_jpg'] + ".jpg"
dirs = os.listdir(source_img)
# print(dirs)

file_list = []
for file in dirs:
    file_list.append(file)
#print (file_list)

matches = annot_data[annot_data['unique_image_jpg'].isin(file_list)]
print("matches with csv: ", len(matches))

for index, row in matches.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = source_img + row['unique_image_jpg']
    print("source path: ", source_path)
    temp1 = cv.imread(source_path, cv.IMREAD_COLOR)
    ################################ good above
    w = row['w'] + 15  # given that x, y are already set back by 10
    h = row['h'] + 15
   # cat1 = row['class']
    #print(cat1)

    xmin_box = row['x_min'] - 20
    if xmin_box < 0:
        xmin_box = 0
    ymin_box = row['y_min'] - 20
    if ymin_box < 0:
        ymin_box = 0
    xmax_box = row['x_min'] + row['w'] + 20
    ymax_box = row['y_min'] + row['h'] + 20
    print(xmin_box, ymin_box, xmax_box, ymax_box)

    # (x, y starting points), (x,y end points)
    cv.rectangle(temp1, (xmin_box, ymin_box), (xmax_box, ymax_box), (0, 255, 0))
    export_path_bird2 = export_path_bird + row['unique_image_jpg']
    print("export path: ", export_path_bird2)

    cv.imwrite(export_path_bird2, temp1, [int(cv.IMWRITE_JPEG_QUALITY), 95])
