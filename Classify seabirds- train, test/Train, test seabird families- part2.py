
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
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
import time

# csv_path = labeled test dataset
csv_test_data = "H:/seabird_family/1_DATASETS/by_random/family_50.50_random_test.csv"
image_folder = "H:/seabird_family/1_DATASETS/model_crops/"
model_path = "H:/seabird_family/1_DATASETS/by_random/family_swin_t_50.50_random_half.pt"

root_output = "H:/seabird_family/1_DATASETS/by_random/results_swin_t_50.50_by_random_May27/"

classification_report = root_output + "_prediction_report.csv"
print(classification_report)
raw_confusion_matrix = root_output + "_raw_conf_matrix.csv"
normalized_confusion_matrix = root_output + "_normal_conf_matrix.csv"
confusion_matrix_png = root_output + "_normal_conf_matrix_pic.png"

if not os.path.exists(root_output):
    os.mkdir(root_output)

start_time = time.time()

###########
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # must print "Using cuda device" to work

class2index = {"Alcidae": 0,
                            "Anatidae": 1, "Ardeidae": 2,
                            "Charadriiformes": 3, "Cygnus": 4, "Gaviidae": 5, "Haematopodidae": 6,
                            "Hydrobatidae": 7, "Laridae": 8, "Pelecanidae": 9,
                            "Phalacrocoracidae": 10, "Podicipedidae": 11,
                            "Procellariidae": 12,
                            "Scoter": 13, "Skimmer": 14,
                            "Sterninae": 15,
                            "Sulidae": 16, "Threskiornithidae": 17, "Artificial": 18, "Unlisted_object": 19
                            }

# load model
model = torch.jit.load(model_path)
model.to(device)

# test dataset
class CustomDataset(torch.utils.data.Dataset):  ## used for custom data loading
    def __init__(self, csv_test_data, image_folder, transform):
        self.annotations = pd.read_csv(csv_test_data)
        self.image_folder = image_folder
        self.transform = transform_test
        self.class2index = {"Alcidae": 0,
                            "Anatidae": 1, "Ardeidae": 2,
                            "Charadriiformes": 3, "Cygnus": 4, "Gaviidae": 5, "Haematopodidae": 6,
                            "Hydrobatidae": 7, "Laridae": 8, "Pelecanidae": 9,
                            "Phalacrocoracidae": 10, "Podicipedidae": 11,
                            "Procellariidae": 12,
                            "Scoter": 13, "Skimmer": 14,
                            "Sterninae": 15,
                            "Sulidae": 16, "Threskiornithidae": 17, "Artificial": 18, "Unlisted_object": 19
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

test_dataset = CustomDataset(csv_test_data=csv_test_data, image_folder=image_folder, transform=transform_test)

y = len(test_dataset)

print("test dataset: ", y)

## Get test predictions
test_loader2 = DataLoader(test_dataset, shuffle=False)

test_pred_list = []
x=0

with torch.no_grad():
    model.eval()
    for images, labels in test_loader2:
        images, labels = images.to(device), labels.to(device)
        images = images.half()
        test_pred = model.forward(images)

        _, test_pred_classes = torch.max(test_pred, dim=1)

        probs = torch.softmax(test_pred, dim=1)
        preds = probs.argmax(dim=1)
        test_pred_list.append(test_pred_classes.cpu().numpy())
        x = x+1
        print("Classified: ", x)

# below are class index predictions
test_pred_list = [a.squeeze().tolist() for a in test_pred_list]

## Classification report
label_truth = []
for images, labels in test_loader2:
    label_truth.append(labels.cpu().numpy())

score = accuracy_score(label_truth, test_pred_list)
bal_score = balanced_accuracy_score(label_truth, test_pred_list)
print("Overall accuracy is: ", score)
print("Balanced accuracy is: ", bal_score)

# label_truth = [a.squeeze().tolist() for a in label_list]

# Read in class indices
class_names = list(class2index.keys())
class_index = list(class2index.values())

end_time = time.time()
duration_min = (end_time - start_time)/60
duration_sec = duration_min*60
duration_hrs = duration_min / 60

print("duration (seconds): ", duration_sec)
print("duration (minutes): ", duration_min)
print("duration (hours): ",duration_hrs)
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