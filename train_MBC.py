from tqdm import tqdm
import torch
import torch.cuda.amp
import torch.optim as optim

# Training model first
train_acc_list = []
train_loss_list = []

scalar = torch.cuda.amp.GradScaler()

def train (epoch, model, train_loader, device, optimizer, criterion):
    print("\nEpoch: %d" % epoch)
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = images.half()

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.forward(images)
            train_loss = criterion(outputs, labels)
            # new below
            scalar.scale(train_loss).backward()
            scalar.step(optimizer)
            scalar.update()
            #train_loss.backward()  ###################
            #optimizer.step()

            running_loss += train_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
      #return train_loss, train_acc

    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    print("Train loss: %.3f | Train Acc: %.3f" % (train_loss, train_acc))
