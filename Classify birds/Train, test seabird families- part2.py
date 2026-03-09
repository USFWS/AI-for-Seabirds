import torch
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import DataLoader
import os
import PIL
import pandas
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

model_path = 'D:/seabird_families/J_2025_April9_seabird_family_swin_s.csv.pt'
csv_test = "D:/seabird_families/seabird_family_test1.csv"
image_folder = "D:/seabird_families/crops/"
model_name = "seabird_families_swin_s_April10"

classification_report = "D:/seabird_families/" + model_name + "_prediction_report .csv"
raw_confusion_matrix = "D:/seabird_families/" + model_name + "_raw_conf_matrix.csv"
normalized_confusion_matrix = "D:/seabird_families/" + model_name + "_normal_conf_matrix.csv"
confusion_matrix_png = "D:/seabird_families/" + model_name + "_normal_conf_matrix_pic.png"

###########
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

class2index = {"Accipitridae": 0, "Alcidae": 1,
                            "Anatidae": 2, "Ardeidae": 3,
                            "Charadriiformes": 4, "Cygnus": 5, "Gaviidae": 6, "Haematopodidae": 7,
                            "Hydrobatidae": 8, "Laridae": 9, "Pelecanidae": 10,
                            "Phalacrocoracidae": 11, "Podicipedidae": 12,
                            "Procellariidae": 13,
                            "Scoter": 14, "Skimmer": 15,
                            "Sterninae": 16,
                            "Sulidae": 17, "Threskiornithidae": 18, "Artificial": 19, "Unlisted_object": 20
                            }

# load model
model = torch.jit.load(model_path)
model.to(device)

# test dataset
class CustomDataset(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_path, image_folder, transform):
        self.annotations = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform_test
        self.class2index = {"Accipitridae": 0, "Alcidae": 1,
                            "Anatidae": 2, "Ardeidae": 3,
                            "Charadriiformes": 4, "Cygnus": 5, "Gaviidae": 6, "Haematopodidae": 7,
                            "Hydrobatidae": 8, "Laridae": 9, "Pelecanidae": 10,
                            "Phalacrocoracidae": 11, "Podicipedidae": 12,
                            "Procellariidae": 13,
                            "Scoter": 14, "Skimmer": 15,
                            "Sterninae": 16,
                            "Sulidae": 17, "Threskiornithidae": 18, "Artificial": 19, "Unlisted_object": 20
                            }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder, self.annotations.iloc[index, 1])  # r, c; col is image name
        image = PIL.Image.open(img_path)
        image = transform_test(image)
        label = self.annotations.iloc[index, 0]
        label = torch.tensor(label)  #################formerly torch.tensor
        return (image, label)  # records the item


transform_test = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=(0.2335, 0.2444, 0.2143), std=(0.1369, 0.1149, 0.1031))
])

test_dataset = CustomDataset(csv_path=csv_test, image_folder=image_folder, transform=transform_test)

y = len(test_dataset)

print("test dataset: ", y)

## Get test predictions
test_loader2 = DataLoader(test_dataset, shuffle=False)

test_pred_list = []

with torch.no_grad():
    model.eval()
    for images, labels in test_loader2:
        images, labels = images.to(device), labels.to(device)
        test_pred = model.forward(images)

        _, test_pred_classes = torch.max(test_pred, dim=1)

        probs = torch.softmax(test_pred, dim=1)
        preds = probs.argmax(dim=1)
        test_pred_list.append(test_pred_classes.cpu().numpy())

# below are class index predictions
test_pred_list = [a.squeeze().tolist() for a in test_pred_list]

## Classification report
label_truth = []
for images, labels in test_loader2:
    label_truth.append(labels.cpu().numpy())

score = accuracy_score(label_truth, test_pred_list)
bal_score = balanced_accuracy_score(label_truth, test_pred_list)
print("Overal accuracy is: ", score)
print("Balanced accuracy is: ", bal_score)

# label_truth = [a.squeeze().tolist() for a in label_list]

# Read in class indices
class_names = list(class2index.keys())
class_index = list(class2index.values())

## Recall, precision stats report
report1 = sklearn.metrics.classification_report(label_truth, test_pred_list, output_dict=True, labels=class_index,
                                                target_names=class_names,
                                                zero_division=False)
classify_report = pandas.DataFrame(report1).transpose()
classify_report.to_csv(classification_report)

# raw numbers for confusion matrix
class_list = list(class2index.keys())
#class_index = list(test_pred_list)

cm = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index)
cm = pandas.DataFrame(cm).transpose()
class_list = list(class2index.keys())
cm.to_csv(raw_confusion_matrix, header= class_list)

# Normalized confusion matrix
report1 = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index, normalize = "true")
df = pandas.DataFrame(report1).transpose()
class_list = list(class2index.keys())
cm.to_csv(normalized_confusion_matrix, header= class_list)

# Plot confusion matrix
label_list = []
for images, labels in test_loader2:
    label_list.append(labels.cpu().numpy())

label_list = [a.squeeze().tolist() for a in label_list]

idx2class = {v: k for k, v in class2index.items()}

confusion_matrix_df = pd.DataFrame(confusion_matrix(label_list, test_pred_list)).rename(columns=idx2class,
                                                                                        index=idx2class)

# sns.set(rc = {'figure.figsize':(16,8)})
plt.rcParams['figure.dpi'] = 200
plt.figure(figsize=(26, 20))
# plt.rcParams['savefig.dpi'] = 300

final1 = sns.heatmap(confusion_matrix_df / numpy.sum(confusion_matrix_df), annot=True,
                     fmt='.0%', cmap='Blues')

final1.figure.savefig(confusion_matrix_png)