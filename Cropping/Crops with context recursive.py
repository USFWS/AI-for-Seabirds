import pandas
import cv2 as cv
import os

##Input: path = dir with parent images; csv_data = detection csv from inference
# export_path_[bird/nonbird/artif] = specify folders for each class
path = "D:/R2_WHCR_2025/1_parent_images/" #JPG_20250122_095400/"
csv_data = pandas.read_csv("D:/R2_WHCR_2025/YOLO10_detection_results_April15.csv")
export_path_bird = "D:/R2_WHCR_2025/3_crops_all_inference/bird_crops_w_context/"
#export_path_nonbird = "D:/detection_crops_for_inference_2024/2024_Jan5_nonbird_crops/"
#export_path_artif = "D:/detection_crops_for_inference_2024/2024_Jan5_artif_crops/"

if not os.path.exists(export_path_bird):
    os.mkdir(export_path_bird)

# csv_data.columns = (['class', 'score', 'xmin', 'ymin', 'w', 'h', 'unique_image_jpg', 'unique_BB'])


for root, dirs, files in os.walk(path):
#for files in dirs:
    path = os.path.join(root, files)
    print(path)
    basename = os.path.splitext(files)[0] + ".jpg"  # take basename (not path) and add .jpg
    print(basename)
    file_list.append(basename)




for index, row in csv_data.iterrows():
    unique_image_jpg = row['unique_image_jpg']
    source = path + row['unique_image_jpg']  # +'.jpg'
    print("Source: ", source)
    temp1 = cv.imread(source, cv.IMREAD_COLOR)  # this is good
    print("ok")

    temp1.shape
    x = row['xmin'] - 400
    if x < 0:
        x = 0
    y = row['ymin'] - 200
    if y < 0:
        y = 0

    w = row['w'] + 800  # given that x, y are already set back by 10
    h = row['h'] + 400
    cat1 = row['class']
    print(cat1)

    xmin_box = row['xmin'] - 10
    if xmin_box < 0:
        xmin_box = 0
    ymin_box = row['ymin'] - 10
    if ymin_box < 0:
        ymin_box = 0
    xmax_box = row['xmin'] + row['w'] + 10
    ymax_box = row['ymin'] + row['h'] + 10
    # print(xmin_box, ymin_box, xmax_box, ymax_box)

    # (x, y starting points), (x,y end points)
    cv.rectangle(temp1, (xmin_box, ymin_box), (xmax_box, ymax_box), (0, 255, 0))
    ############# INPUT HERE
    if cat1 == "bird":
        crops = temp1[y:(y + h), x:(x + w)]
        cv.imwrite(export_path_bird + row['unique_BB'] + '.jpg', crops, [int(cv.IMWRITE_JPEG_QUALITY), 95])
