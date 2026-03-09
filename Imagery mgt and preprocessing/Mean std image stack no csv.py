
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
image_folder = "D:/SACR/SACR_FIX/old_model_2023_images_gr5_cranes/"

# To tensor, normalize based on our data
transform1 = transforms.Compose([
    transforms.ToTensor()
])

# Create train dataset
train_dataset = (image_folder == image_folder, transform= transform1)

len(train_dataset)

## Stacks all tensors together in single batch
stack1 = torch.stack([image for image, _ in train_dataset], dim = 3)
stack1.shape

# Merges into a 3 x (224*224) vector for each channel, then computes the mean
x = stack1.view(1,-1).mean(dim=1)
y = stack1.view(1,-1).std (dim=1)
print(x, y)



