import pandas as pd
import os
import shutil
import config

# The script searches a image directory and copies/moves images that are listed in a csv file
# the csv file should have column labeled as "unique_image_jpg"

# Data inputs:
# csv data= list of images to be copied/moved (column header of 'unique_image_jpg')
# source_img = directory of images to search
# dest1 = destination folder to move/copy images into

source_img = config.SOURCE_IMG

csv_data = pd.read_csv(config.CSV_DATA)

export_dir = config.EXPORT_DIR

if not os.path.exists(export_dir):
    os.mkdir(export_dir)

##if jpg is needed run this:
csv_data['unique_image_jpg'] = csv_data['unique_image_jpg'] #+ ".png"

x = 0

csv_list = []
csv_list = csv_data['unique_image_jpg'].tolist()
print(csv_list)

for root, dirs, files in os.walk(source_img):
    for filename in files:
        if filename in csv_list:
            x = x + 1
            print ("Copied: ", x)
            path = os.path.join(root, filename)
            print ("path " , path)
            shutil.copy(path, export_dir)  # can be shutil.move or shutil.copy
        else:
            pass