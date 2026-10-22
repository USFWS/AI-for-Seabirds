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
flight_folder = "C:/users/aware/desktop/flight_folder/"
infer_crops = "infer_bird_crops"
sampling = "yes"
sample_proportion = 0.10

new_csv = flight_folder + "/classify_" + infer_crops  + ".csv"
root_export = flight_folder + "/classification_results/"
image_dir = flight_folder + infer_crops  + "/"
image_context = flight_folder + "/crops_w_context_birds/"

model_path = 'C:/users/aware/desktop/MODELS FOR USE/2025_April10_seabird_family_swin_s.pt'

prob_threshold = 1.00

if not os.path.exists(root_export):
    os.mkdir(root_export)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

# load model
model = torch.jit.load(model_path)
model.to(device)

transform_test = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize(mean= (0.2335, 0.2444, 0.2143), std=(0.1369,0.1149, 0.1031))
])

idx_to_label = {0: "Accipitridae", 1: "Alcidae",
                            2: "Anatidae", 3: "Ardeidae",
                            4: "Charadriiformes", 5: "Cygnus", 6: "Gaviidae", 7: "Haematopodidae",
                            8: "Hydrobatidae", 9: "Laridae", 10: "Pelecanidae",
                            11: "Phalacrocoracidae", 12: "Podicipedidae",
                            13: "Procellariidae",
                            14: "Scoter", 15: "Skimmer",
                            16: "Sterninae",
                            17: "Sulidae", 18: "Threskiornithidae", 19: "Artificial", 20: "Unlisted_object"
                            }

species_list = list(idx_to_label.values())

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', 'label1', 'label2',  'score1', 'score2'])

def classify(model, transform_test, source):
    model = model.eval()
    image = PIL.Image.open(source)
    image = transform_test(image).float()
    image = image.to(device)
    image = image.unsqueeze(0)
    output = model(image)
    # print(output.data)
    softmax = torch.nn.functional.softmax(output, dim=1)

    top3_prob, top3_label = torch.topk(softmax, 3)
    # print("tops: ", top3_prob,top3_label)
    label1 = top3_label[0, 0]
    label2 = top3_label[0, 1]
    score1 = top3_prob[0, 0]
    score2 = top3_prob[0, 1]
    label1 = label1.data.cpu().numpy()
    label2 = label2.data.cpu().numpy()
    print(label1, label2)
    if label1 > 20 or label2 > 20:
        label1 = 20
        label2 = 20

    score1 = score1.data.cpu().numpy()
    score2 = score2.data.cpu().numpy()
    species_list = list(idx_to_label.values())

    label1 = species_list[label1]
    label2 = species_list[label2]

    print(label1, label2)

    with open(new_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, label1, label2, score1, score2])

for root, dirs, files in os.walk(image_dir):
    for file in files:
        source = os.path.join(root, file)
        name = os.path.basename(source)
        #print ("name: ", name)
        classify(model, transform_test, source)
    else:
        pass

# This part does the moving
dirs = os.listdir(image_context)  # get all files in folder

csv_data = pandas.read_csv(new_csv)

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
        print("Too high!")

if sampling == 'yes':
    print("Subsampling now!")
    for dir, subfolders, files in os.walk(root_export):
        for subfolder in subfolders:
            #print (subfolder)
            file_count = len(subfolder)
            print(subfolder, file_count)
            sample_size = sample_proportion * file_count
            sample_size = round(sample_size)

            for i in range(sample_size):
                sub_name = root_export + subfolder
                random_img = random.choice(os.listdir(sub_name))
                random_source =  root_export + subfolder + "/" + random_img
                random_export = root_export + subfolder + "_random/"
                if not os.path.exists(random_export):
                    os.mkdir(random_export)
                random_export2 = random_export + random_img
               # print ("random export: ", random_export2)
                shutil.copy(random_source, random_export2)
