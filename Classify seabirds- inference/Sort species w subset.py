import torch
import PIL
from PIL import Image
from torchvision import transforms
import csv
from os.path import basename
import os
import shutil
import pandas
import random

# Inputs:
# root_path = drive
# flight_name

# image_dir = directory of images to apply inference to
# root_export = directory where species folders are set up
# Optional (if new model is applied): idx_to_label = index to the corresponding label in the model
# model_path = pytorch classification model saved as script file
# transform_test = transform to be applied prior to inference

## New inputs: drive_path = root directory, flight_name = flight folder, model_path = model to apply
project_folder = "E:/seabirds_winter/"
classify_csv =  "E:/seabirds_winter/classify_yolo_feb2025_merged_crops.csv"
root_export = project_folder + "/classification_results2/"
image_context = project_folder + "/yolo_feb2025_merged_crops_w_context/"

sampling = "yes"
sample = 100
prob_threshold = 0.30

if not os.path.exists(root_export):
    os.mkdir(root_export)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

# This part does the moving
dirs = os.listdir(image_context)  # get all files in folder

csv_data = pandas.read_csv(classify_csv)

for index, row in csv_data.iterrows():
    score1 = row['score1']
    print(score1)

    if score1 <= prob_threshold:
        target = image_context + row['unique_image_jpg']  # +'.jpg'
        name = basename(target)
        dest1 = root_export + "/unknown/"
        if not os.path.exists(dest1):
            os.makedirs(dest1)

        dest2 = root_export + "unknown/" + name
        print ('new ; ', dest2)
        shutil.copy(target, dest2)

    else:
        if score1 >  prob_threshold:
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
            print("Too high!")

if sampling == 'yes':
    print("Subsampling now!")
    random_export = os.path.join(root_export, "random_subset") + "/"
    if not os.path.exists(random_export):
        os.mkdir(random_export)

        # only look at top-level species folders, skip the random_subset folder itself
    species_folders = [
        d for d in os.listdir(root_export)
        if os.path.isdir(os.path.join(root_export, d)) and d != "random_subset"
        ]

    for subfolder in species_folders:
        sub_name = os.path.join(root_export, subfolder)
        files_in_folder = [
            f for f in os.listdir(sub_name)
            if os.path.isfile(os.path.join(sub_name, f))
            ]
        file_count = len(files_in_folder)
        print(subfolder, file_count)

        sample_size = min(sample, file_count)
        chosen_files = random.sample(files_in_folder, sample_size)

        for random_img in chosen_files:
            random_source = os.path.join(sub_name, random_img)
            random_export2 = os.path.join(random_export, random_img)
            shutil.copy(random_source, random_export2)