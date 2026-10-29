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
infer_crops = "yolo_feb2025_forward_crops"
sampling = "yes"
sample = 100

new_csv = project_folder + "/classify_" + infer_crops + ".csv"
root_export = project_folder + "/classification_results_weighted/"
image_dir = project_folder + infer_crops + "/"
image_context = project_folder + "/yolo_feb2025_forward_crops_w_context/"

model_path = 'E:/seabirds_winter/species_swin_s_70.30_weighted_Oct27.pt'
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

idx_to_label = {"ABDU": 0, "SATE": 1,
                            "AMOY": 2, "artificial": 3,
                            "BCPE": 4, "BLGU": 5, "BLSC": 6,
                            "BLSK": 7, "BOGU": 8,
                            "BRAN": 9, "BRPE": 10,
                            "BUFF": 11, "CANG": 12, "COEI": 13,
                            "COLO": 14, "Cormorant": 15, "CORS": 16,
                            "COTE": 17, "GBBG": 18,
                            "GREG": 19, "GRSH": 20, "HERG_RBGU": 21,
                            "HOGR": 22, "LAGU": 23,
                            "COGO": 24,
                            "LTDU": 25, "MALL": 26, "Mergus": 27,
                            "NOGA": 28, "Phalarope": 29, "RAZO": 30,
                            "REDH": 31, "RNGR": 32,
                            "COMU": 33,
                            "ROYT": 34,
                            "RTLO": 35, "SCAUP": 36, "Shorebirds": 37,
                            "SNGO": 38, "SOSH": 39, "SUSC":40, "TUSW": 41,
                            "unlisted_object": 42, "WHIB": 43, "WISP": 44,
                            "WWSC": 45, "FOTE": 46
                            }

# index -> label name lookup (idx_to_label is label name -> index)
label_list = list(idx_to_label.keys())

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', 'label1', 'label2',  'score1', 'score2'])

def classify(model, transform_test, source):
    model = model.eval()
    image = PIL.Image.open(source)
    image = transform_test(image).float()
    image = image.to(device)
    image = image.unsqueeze(0)
    image = image.half()
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
   # print(label1, label2)

    score1 = score1.data.cpu().numpy()
    score2 = score2.data.cpu().numpy()

    label1 = label_list[label1]
    label2 = label_list[label2]
    print(label1, label2)

    with open(new_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, label1, label2, score1, score2])
x = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        source = os.path.join(root, file)
        name = os.path.basename(source)
        x = x + 1
        print(x)
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
        if sub_name == "unlisted_object":
            pass
        if sub_name == "artificial":
            pass

        random_source = os.path.join(sub_name, random_export)
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