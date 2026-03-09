import pandas
import cv2 as cv

annotations = "D:/detection_of_seabirds/not_reviewed_images/tiles_annot_no_empties.csv"
image_path = "D:/detection_of_seabirds/not_reviewed_images/tiles_no_empties_10972_boxes/"
export_path = "D:/detection_of_seabirds/not_reviewed_images/tiles_no_empties_10972_boxes/"

annotations = pandas.read_csv(annotations)
annotations.columns = (['image_id', 'xmin', 'ymin', 'w', 'h', 'label_id', 'unique_image_jpg'])

for index, row in annotations.iterrows():  ## iterrows: Pandas iterate over rows
    source_path = image_path + row['unique_image_jpg']
    input = cv.imread(source_path, cv.IMREAD_COLOR)  # this
    print("source: ", source_path)
   # check = os.path.exists(source_path)

    xmin = row['xmin'] - 22
    ymin = row['ymin'] - 22
    w = row['w'] + 44
    h = row['h'] + 44
    xmax = xmin + w
    ymax = ymin +h

    cv.rectangle (input, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    new_name = export_path + row['unique_image_jpg']
    print("new name: ", new_name)
    cv.imwrite(new_name, input, [int(cv.IMWRITE_JPEG_QUALITY), 95])
