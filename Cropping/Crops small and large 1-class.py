import pandas
import cv2 as cv
import os

##Inputs: 
# source_path = dir with parent images
# csv_data = detection model results in csv
# export_path_context = export directory for crops with context 
# export_path_small = export directory for small crops 

source_path = "D:/2025/1_parent_images/JPG_20250122_095400/"
csv_data = pandas.read_csv("D:/2025/12_detection/g_inference/yolov11s_Aug3_095400.csv")
export_path_context = "D:/2025/12_detection/g_inference/crops_context_095400/"
export_path_small = "D:/2025/12_detection/g_inference/crops_095400/"

if not os.path.exists(export_path_context):
    os.mkdir(export_path_context)
if not os.path.exists(export_path_small):
    os.mkdir(export_path_small)

#csv_data.columns = (['unique_image_jpg', 'class', 'score', 'xmin', 'ymin', 'w', 'h', 'unique_BB'])

dirs = os.listdir(source_path)  # get all files in folder
print("image path: ", len(dirs))

# Get all of the image names without the path
file_list = []
for file in dirs:
    basename = os.path.splitext(file)[0] # + ".jpg"  # take basename (not path) and add .jpg
   # print(basename)
    file_list.append(basename)

matches = csv_data[csv_data['unique_image_jpg'].isin(file_list)]
print("matches with csv: ", len(matches))

for index, row in matches.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = source_path + row['unique_image_jpg']  # +'.jpg'
    print("Source path: ", source_path)
    temp1 = cv.imread(source_path, cv.IMREAD_COLOR)  # this is good

    temp1.shape
    x = row['xmin'] - 400
    if x < 0:
        x = 0
    y = row['ymin'] - 200
    if y < 0:
        y = 0

    w = row['w'] + 800  # given that x, y are already set back by 10
    h = row['h'] + 400
    # cat1 = row['class']

    xmin_box = row['xmin'] - 10
    if xmin_box < 0:
        xmin_box = 0
    ymin_box = row['ymin'] - 10
    if ymin_box < 0:
        ymin_box = 0
    xmax_box = row['xmin'] + row['w'] + 10
    ymax_box = row['ymin'] + row['h'] + 10
    #print(xmin_box, ymin_box, xmax_box, ymax_box)

    # (x, y starting points), (x,y end points)
    cv.rectangle(temp1, (xmin_box, ymin_box), (xmax_box, ymax_box), (0, 255, 0))

    export_path = export_path_context
    crops = temp1[y:(y + h), x:(x + w)]
    name = export_path_context + row['unique_BB'] + ".jpg"
    print ("dest1", name)

  #  cv.imwrite(name, crops, [int(cv.IMWRITE_JPEG_QUALITY), 95])

    ## Crops for inference
   # new_path2 = path + row['unique_image_jpg']  # +'.jpg'
    #temp1 = cv.imread(new_path2, cv.IMREAD_COLOR)  # this is good
    temp2 = cv.imread(source_path, cv.IMREAD_COLOR)
    temp2.shape
    x = row['xmin'] - 10
    if x < 0:
        x = 0
    y = row['ymin'] - 10
    if y < 0:
        y = 0

    #cat1 = row['class']
    #print(cat1)
    w = row['w'] + 20
    h = row['h'] + 20
    print(xmin_box, ymin_box, xmax_box, ymax_box)

    # Specify each class name below (cat1, cat2, etc.)

    export_path = export_path_small
    crops = temp2[y:(y + h), x:(x + w)]
    name2 = row['unique_BB'] + ".jpg"
    print("dest2", name2)
    cv.imwrite(export_path + name2 , crops, [int(cv.IMWRITE_JPEG_QUALITY), 95])



