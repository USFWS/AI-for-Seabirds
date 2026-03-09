import seaborn as sns
import pandas
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import numpy
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

class2index = {"ATPU": 0, "BCPE": 1,
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
df = pandas.DataFrame(report1).transpose()
df.to_csv("D:/2024_Nov20_focal_nov20_report_swin_s.csv")

# Raw number confusion matrix
# Read in class indices
class_list = list(class2index.keys())
#class_index = list(test_pred_list)

cm = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index)
cm = pandas.DataFrame(cm).transpose()
class_list = list(class2index.keys())

cm.to_csv("D:/2024_Nov20_focal_augm15_conf-matrix_swin_s.csv", header= class_list)

# normalized confusion matrix
## conf matrix works with index labels
report1 = sklearn.metrics.confusion_matrix(label_truth, test_pred_list, labels = class_index, normalize = "true")
df = pandas.DataFrame(report1).transpose()
class_list = list(class2index.keys())

df.to_csv("D:/2024_Nov20_focal_aug15m_conf_matrix_norm_swin_s.csv", header = class_list)

# plot confusion matrix
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

final1.figure.savefig("D:/AMAPPS_CLASSIFY/confusion_matrix10.png")

score = accuracy_score(label_truth, test_pred_list)
bal_score = balanced_accuracy_score(label_truth, test_pred_list)
print("Overal accuracy is: ", score)
print("Balanced accuracy is: ", bal_score)