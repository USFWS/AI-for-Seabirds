import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import csv
import time
import config
import pandas
import shutil

image_source = config.SOURCE_IMG
root_export_dir = config.ROOT_EXPORT_DIR

model_path = config.MODEL_PATH_CLASSIFY
crops_w_context = config.CROPS_W_CONTEXT

prob_threshold = 1.0
batch_size = 66

new_csv = root_export_dir + "classify_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', 'label1', 'label2', 'label3', 'score1', 'score2', 'score3'])

# load model
model = torch.jit.load(model_path)
model.half()
model.to(device)

time_start = time.time()

idx_to_label = {0: "Alcidae",
                            1: "Anatidae", 2: "Ardeidae",
                            3: "Charadriiformes", 4: "Cygnus", 5: "Gaviidae", 6: "Haematopodidae",
                            7: "Hydrobatidae", 8: "Laridae", 9: "Pelecanidae",
                            10: "Phalacrocoracidae", 11: "Podicipedidae",
                            12: "Procellariidae",
                            13: "Scoter", 14: "Skimmer",
                            15: "Sterninae",
                            16: "Sulidae", 17: "Threskiornithidae", 18: "Artificial",
                19: "Unlisted_object"
                            }
species_list = list(idx_to_label.values())
print(species_list)

class ImageNameDataset(Dataset):
    """
    Custom PyTorch Dataset that returns:
    - image tensor
    - image filename (without path)
    """

    def __init__(self, image_dir, transform=None):
        if not os.path.isdir(image_dir):
            raise ValueError(f"Provided path '{image_dir}' is not a valid directory.")

        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return both tensor and filename

# Example usage for inference
if __name__ == "__main__":
    # Define transforms (resize + tensor conversion)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.2335, 0.2444, 0.2143), std=(0.1369, 0.1149, 0.1031))
    ])

    dataset = ImageNameDataset(image_source, transform=transform)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=False)

    with torch.no_grad():
        for images, names in dataloader:
          images = images.to(device)
          images = images.half()
          img_name = names
          #[row[0] for row in t]
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          softmax = torch.nn.functional.softmax(outputs, dim=1)
          top3_prob, top3_label = torch.topk(softmax, 3)
         # print(img_name)

          top3_label = top3_label.cpu()
          top3_label = top3_label.numpy()
          top3_prob = top3_prob.cpu()
          top3_prob = top3_prob.numpy()
## new
          score1 = [row[0] for row in top3_prob]
          score2 = [row[1] for row in top3_prob]
          score3 = [row[2] for row in top3_prob]
          score1 = np.array(score1)
          score2 = np.array(score2)
          score3 = np.array(score3)

          label1 = [row[0] for row in top3_label]
          label2 = [row[1] for row in top3_label]
          label3 = [row[2] for row in top3_label]
          label1 = np.array(label1)
          label2 = np.array(label2)
          label3 = np.array(label3)

          img_name = [row for row in img_name]

          print("Labels: ", label1, label2, label3)
          print("Scores: ", score1, score2, score3)
          print(img_name)

          current_batch_size = len(img_name)
          print("batch size is:", current_batch_size)

          with open(new_csv, 'a', newline='') as file:
            writer = csv.writer(file)
            for i in range(0, current_batch_size):
                score1a = score1[i]
                score2a = score2[i]
                score3a = score3[i]
                label1a = label1[i]
                label1a = species_list[label1a]
                label2a = label2[i]
                label2a = species_list[label2a]
                label3a = label3[i]
                label3a = species_list[label3a]
                unique_BB = img_name[i]

                print("correct", unique_BB, label1a, label2a, label3a, score1a, score2a, score3a)
                writer.writerow([unique_BB, label1a, label2a, label3a, score1a, score2a, score3a])

time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Model completed in, seconds:", duration_sec)
print("Model completed in, minutes:", duration_min)

# This part does the moving
dirs = os.listdir(crops_w_context)  # get all files in folder
csv_data = pandas.read_csv(new_csv)

for index, row in csv_data.iterrows():
    score1 = row['score1']
    print(score1)
    if score1 <  prob_threshold:
        target = crops_w_context + row['unique_image_jpg']  # +'.jpg'
        print('Target : ', target)
        cat1 = row['label1']
        print("Class: ", cat1)

        for folders, subfolders, files in os.walk(crops_w_context):
            name = os.path.basename(target)
            print ("name: ", name)
            if name in files:
                dir2 = root_export_dir + row['label1']
                if not os.path.exists(dir2):
                    os.makedirs(dir2)
                dest = root_export_dir + row['label1'] + '/' + name

                print ("Destination : ", dest)
                shutil.copy(target, dest)  # this can be changed to: shutil.move
            else:
                pass
    else:
        print("Threshold is too high!")