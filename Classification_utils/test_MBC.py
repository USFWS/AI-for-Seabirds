import torch
import torch.cuda.amp
from torch.cpu.amp import autocast

val_acc_list = []
val_losses_list = []

# scalar = torch.cuda.amp.GradScaler()
best_val_loss = float('inf')
counter = 0

def test(model: object, test_loader: object, device: object, criterion: object) -> None:
    model.to(device)
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
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
    torch.cuda.empty_cache()





