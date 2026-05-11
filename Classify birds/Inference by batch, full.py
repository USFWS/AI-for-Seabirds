import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import csv
import time

batch_size= 67

image_source = main_dir = "E:/R2_waterfowl/timer_img/demo/"
new_csv = "E:/R2_waterfowl/timer_img/practice.csv"

model_path = "E:/R2_waterfowl/model_weights/2026_winter_waterfowl_swin_s_scripted1.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

with open(new_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['unique_image_jpg', 'label1', 'label2', 'label3', 'score1', 'score2', 'score3'])

# load model
model = torch.jit.load(model_path)
model.to(device)

time_start = time.time()

idx_to_label = {0: "BWTE", 1: "CANV", 2: "COGO", 3: "grebe",
        4: "raptor", 5: "RBME", 6: "AGWT",
        7: "AMAV", 8: "AMWI", 9: "artificial",
        10: "AWPE", 11: "BRPE", 12: "BUFF", 13: "CANG",
        14: "Cormorant", 15: "GADW", 16: "gull",
        17: "MALL", 18: "NOPI", 19: "NSHO", 20: "REDH", 21: "ROSP",
        22: "SACR", 23: "SCAU", 24: "shorebird",
        25: "SNGO", 26: "tern", 27: "unlisted_object",
        28: "wading_bird", 29: "WHIB", 30: "white_egret",
        31: "WHCR_adult", 32: "WHCR_juvenile"
        }
unlisted_object_index = 9
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

          species_list = list(idx_to_label.values())

          label1 = [row[0] for row in top3_label]
          label2 = [row[1] for row in top3_label]
          label3 = [row[2] for row in top3_label]
          label1 = np.array(label1)
          label2 = np.array(label2)
          label3 = np.array(label3)
          #

          img_name = [row for row in img_name]

          print("Labels: ", label1, label2, label3)
          print("Scores: ", score1, score2, score3)
          print(img_name)

          print("batch size is:", batch_size)

          for i in range(0, batch_size):
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
              # img_name = img_name[2]

              print("corect", unique_BB, label1a, label2a, label3a, score1a, score2a, score3a)
              with open(new_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([unique_BB, label1a, label2a, label3a, score1a, score2a, score3a])

time_end = time.time()
duration_sec = time_end - time_start
duration_min = (duration_sec / 60)
duration_hrs = (duration_min / 60)

print("Completed in, seconds:", duration_sec)
print("Completed in, minutes:", duration_min)
print("Completed in, hours:", duration_hrs)
