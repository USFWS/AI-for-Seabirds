import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import pandas as pd
import os
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
import torch.utils
from torchvision import transforms, models
from torch.optim import lr_scheduler
torch.set_printoptions(edgeitems=2)
# torch.manual_seed(123)
import numpy as np
import Classification_utils
from Classification_utils import train_MBC, test_MBC


device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_cuda = torch.cuda.is_available()
if device:
    print(torch.cuda.get_device_name())

# Inputs
csv_train = "D:/seabird_families/seabird_family_train1.csv"
csv_test = "D:/seabird_families/seabird_family_test1.csv"
image_folder = "D:/seabird_families/crops/"

# Valdiation model 2nd
val_losses = []
val_acc = []

# Train dataset
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(2, 15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2335, 0.2444, 0.2143), std=(0.1369, 0.1149, 0.1031))
])

class CustomTrain(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_path, image_folder, transform):
        self.annotations = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform_train
        self.class2index = {"ATPU": 0, "BCPE": 1,
                            "BLSC": 2, "BOGU": 3,
                            "BRPE": 4, "BUFF": 5, "CANG": 6, "COEI": 7,
                            "COGO": 8, "COLO": 9, "DCCO": 10,
                            "GBBG": 11, "HERG": 12,
                            "HOGR": 13,
                            "LAGU": 14, "LTDU": 15,
                            "NOGA": 16,
                            "not_wildlife": 17, "RBME": 18, "REDH": 19,
                            "ROYT": 20, "RTLO": 21, "SCAU": 22, "SNGO": 23, "SUSC": 24,
                            "TUSW": 25, "WWSC": 26
                            }
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.annotations.iloc[index, 1])  # r, c; col is image name
        image = PIL.Image.open(img_path)
        image = transform_train(image)
        label = self.annotations.iloc[index, 0]
        label = torch.tensor(label)  #################formerly torch.tensor
        return image, label  # records the item

train_dataset = CustomTrain(csv_path=csv_train, image_folder=image_folder, transform=transform_train)

x = len(train_dataset)
print("Train dataset; ", x)

class CustomDataset(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_path, image_folder, transform):
        self.annotations = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform_test
        self.class2index = {"ATPU": 0, "BCPE": 1,
                            "BLSC": 2, "BOGU": 3,
                            "BRPE": 4, "BUFF": 5, "CANG": 6, "COEI": 7,
                            "COGO": 8, "COLO": 9, "DCCO": 10,
                            "GBBG": 11, "HERG": 12,
                            "HOGR": 13,
                            "LAGU": 14, "LTDU": 15,
                            "NOGA": 16,
                            "not_wildlife": 17, "RBME": 18, "REDH": 19,
                            "ROYT": 20, "RTLO": 21, "SCAU": 22, "SNGO": 23, "SUSC": 24,
                            "TUSW": 25, "WWSC": 26
                            }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.annotations.iloc[index, 1])  # r, c; col is image name
        image = PIL.Image.open(img_path)
        image = transform_test(image)
        label = self.annotations.iloc[index, 0]
        label = torch.tensor(label)  #################formerly torch.tensor
        return image, label  # records the item

transform_test = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=(0.2335, 0.2444, 0.2143), std=(0.1369, 0.1149, 0.1031))
])

test_dataset = CustomDataset(csv_path=csv_test, image_folder=image_folder, transform=transform_test)

y = len(test_dataset)
print("Test dataset: ", y)

class_counts = np.bincount(train_dataset.annotations.species_cat)
print ("class counts: ", class_counts)
num_classes = len(class_counts)
total_samples = len(train_dataset.annotations.species_cat)
class_weights = []

for count in class_counts:
    weight = 1 / (count / total_samples)
    class_weights.append(weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets]*(1-pt) ** self.gamma* ce_loss).mean()
        return loss

class_weights = torch.FloatTensor(class_weights)
class_weights = class_weights.to(device)
print("Class weights: ", class_weights)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load a pretrained transformer model
# Available models: swin_s, swin_t
#model = torch.load ("I:/Saved_models_weights/swin_s.pth")
from torchvision.models import swin_s, swin_t, swin_b, maxvit_t, maxvit, swin_v2_t, swin_v2_s

## Swin models
weights = torchvision.models.Swin_S_Weights.IMAGENET1K_V1
model = models.swin_s(weights= weights)

## Swin models v2
#weights = torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1
#model = models.swin_v2_t(weights= weights)

# Maxvit_t
#weights = torchvision.models.MaxVit_T_Weights.IMAGENET1K_V1
#model = models.maxvit_t (weights= weights)

# Vit
#weights = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
#model = models.vit_b_32(weights= weights)

# You can also use `weights=Swin_T_Weights.DEFAULT` to get the most up-to-date weights.

for param in model.parameters():
    param.requires_grad = True

# Add fully connect layer to model
model.fc = nn.Sequential(# nn.Linear(2048,512), ##Resent50 only
                        # nn.ReLU(),
                         #nn.Dropout(0.5),
                         nn.Linear(1000,500),
                         nn.ReLU(),
                         nn.Linear(250,125),
                         nn.ReLU(),
                         nn.Linear(125,34),
                         nn.Softmax(dim=1))
model.to(device)
################
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=class_weights, gamma=2)
n_epoch = 8
#optimizer = optim.SGD(model.parameters(),lr= 0.001, momentum = 0.90) ##0.001 good start
optimizer = optim.Adam (model.parameters(), lr= 0.00001)

for epoch in range(1,n_epoch+1):
    Classification_utils.train_MBC.train(epoch, model, train_loader, device=device, optimizer=optimizer, criterion=criterion)
    Classification_utils.test_MBC.test(epoch, model, test_loader, device=device)

# export model and weights
model_scripted = torch.jit.script(model)
model_scripted.save('C:/users/aware/desktop/2025_Apri9_seabird_family_swin_s_scripted1.pt')