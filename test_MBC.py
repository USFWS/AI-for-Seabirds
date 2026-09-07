import torch
import torch.cuda.amp
from torch.cpu.amp import autocast
import Classification_utils
from Classification_utils import train_MBC, test_MBC
from Classification_utils import model_plots

from tqdm import tqdm
val_acc_list = []
val_losses_list = []

scalar = torch.cuda.amp.GradScaler()

def test(epoch, model, test_loader, device, optimizer, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with autocast():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model.forward(images)

                val_loss = criterion(outputs, labels)
                running_loss += val_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            #   early_stopping = EarlyStopping(tolerance=0.3, min_delta=0.3) ### early stopping, as needed

    val_loss = running_loss / len(test_loader)
    val_acc = 100. * correct / total

    val_losses_list.append(val_loss)
    val_acc_list.append(val_acc)

    print('Val Loss: %.3f | Val Acc: %.3f' % (val_loss, val_acc))





    # early_stopping = EarlyStopping(tolerance =3, min_delta = 0.3)