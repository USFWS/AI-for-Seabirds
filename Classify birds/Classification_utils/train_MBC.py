from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
# Training model first
train_acc = []
train_losses = []

def train (epoch, model, train_loader, device, optimizer, criterion):
    print("\nEpoch: %d" % epoch)
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model.forward(images)
        train_loss = criterion(outputs, labels)

        train_loss.backward()  ###################
        optimizer.step()

        running_loss += train_loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    acc = 100. * correct / total

    train_acc.append(acc)
    train_losses.append(train_loss)
    print("Train loss: %.3f | Train Acc: %.3f" % (train_loss, acc))
