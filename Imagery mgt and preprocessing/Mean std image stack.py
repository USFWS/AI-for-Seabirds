
import PIL
import pandas as pd
import os
from tqdm.auto import tqdm
import torch.utils.data
from torchvision import transforms
import torch
torch.set_printoptions (edgeitems=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

use_cuda = torch.cuda.is_available()
if device:
    print(torch.cuda.get_device_name())

# Inputs
csv_train = "D:/SACR_models/2023_sacr/two_step_dataset/train_export.csv"
image_folder = "D:/SACR_models/2023_sacr/two_step_dataset/all_crops/"

class CustomDataset(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_path, image_folder, transform):
        self.annotations = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform1
        self.class2index = {"ducks": 0, "sacr": 1}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.annotations.iloc[index, 1])  # r, c; col is image name
        image = PIL.Image.open(img_path)
        image = transform1(image)
        label = self.annotations.iloc[index, 0]
        label = torch.tensor(label)  #################formerly torch.tensor
        return (image, label)  # records the item

# To tensor, normalize based on our data
transform1 = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

# Create train dataset
train_dataset = CustomDataset(csv_path = csv_train, image_folder = image_folder, transform= transform1)

len(train_dataset)

## Stacks all tensors together in single batch
stack1 = torch.stack([image for image, _ in train_dataset], dim = 3)
stack1.shape

# Merges into a 3 x (224*224) vector for each channel, then computes the mean
x = stack1.view(1,-1).mean(dim=1)
y = stack1.view(1,-1).std (dim=1)
print(x, y)



