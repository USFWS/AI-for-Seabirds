import os
import pandas
from os.path import basename
import shutil

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
root_path = "D:/classify_seabirds/"
flight_name = "inference_2024"

new_csv = root_path + flight_name + "/classify_" + flight_name + ".csv"
root_export = root_path + flight_name + "/classification_results/"
image_dir = root_path + flight_name + "/2024_bird_crops_inference/"
image_context = root_path + flight_name + "/2024_crops_w_context_birds/"

prob_threshold = 1.0

dirs = os.listdir(image_dir)  # get all files in folder

csv_data = pandas.read_csv(new_csv)
print(csv_data)

for index, row in csv_data.iterrows():
    score1 = row['score1']
    print(score1)
    if score1 <  prob_threshold:
        print("Okay")
        target = image_context + row['unique_image_jpg']  # +'.jpg'
        print('Target : ', target)
        cat1 = row['label1']
        print("Class: ", cat1)

        for folders, subfolders, files in os.walk(image_context):
            name = basename(target)
            print ("name: ", name)
            if name in files:
                dir2 = root_export + row['label1']
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                dest = root_export + row['label1'] + '/' + name

                print ("Destination : ", dest)
                shutil.copy(target, dest)  # this can be changed to: shutil.move
            else:
                pass
    else:
        print("Too high!")
