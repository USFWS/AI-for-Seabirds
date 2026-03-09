
import os
import shutil
import random

flight_folder = "C:/users/aware/desktop/flight_folder"
infer_crops = "infer_bird_crops"
sampling = "yes"
sample_proportion = 0.20

new_csv = flight_folder + "/classify_" + infer_crops  + ".csv"
root_export = flight_folder + "/classification_results/"
image_dir = flight_folder + infer_crops  + "/"
image_context = flight_folder + "/crops_w_context_birds/"

subfolder_list = []
file_count_list = []

if sampling == 'yes':
    print("Subsampling now!")
    for filename in os.listdir(root_export):
        print(filename)
        subfolder_path = root_export + (filename)
        print(subfolder_path)
        subfolder_list.append(subfolder_path)
        x = 0

        for file in os.listdir(subfolder_path):
            print(file)
            x = x + 1
            print(x)
        file_count_list.append(x)

print (subfolder_list)
print(file_count_list)

for subfolder in subfolder_list:
    print(subfolder)
    random_img = random.choice(os.listdir(subfolder))
    random_source = subfolder + "/" + random_img
    print("random source: ", random_source)
    random_export1 = subfolder + "_random/"
    if not os.path.exists(random_export1):
        os.mkdir(random_export1)
    random_export2 = random_export1 + random_img
    print("random export: ", random_export2)
    shutil.copy(random_source, random_export2)

