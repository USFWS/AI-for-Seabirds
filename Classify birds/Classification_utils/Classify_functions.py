import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
import pandas as pd
import os
from tqdm.notebook import tqdm as tqdm
import torch.utils.data
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

class classify_birds:

    def train(n_epoch):
        print(\\Epoch: %d\ %epoch),
        model.train(),
        running_loss = 0,
        correct = 0,
        total = 0,
        train_acc = [],
        train_losses = [],

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device),
            optimizer.zero_grad(),
            outputs = model.forward(images),
            train_loss = criterion(outputs, labels),
            train_loss.backward(),
            optimizer.step(),
            running_loss += train_loss.item(),
            _, predicted = outputs.max(1),
            total += labels.size(0),
            correct += predicted.eq(labels).sum().item(),
            train_loss = running_loss / len(train_loader),
            acc = 100. * correct / total,
            train_acc.append(acc),
            train_losses.append(train_loss),
            print(\Train
            loss: %.3
            f | Train
            Acc: %.3
            f\ %(train_loss, acc))

            # Valdiation model 2nd,
            val_losses = [],
            val_acc = [],
            ,

            def test(epoch):
                ,

            model.eval(),
            running_loss = 0,
            correct = 0,
            total = 0,
            ,
            with torch.no_grad():
                ,
            for images, labels in test_loader: ,
            images, labels = images.to(device), labels.to(device),
        ,
        outputs = model.forward(images),
        val_loss = criterion(outputs, labels),
        running_loss += val_loss.item(),
        _, predicted = outputs.max(1),
        total += labels.size(0),
        correct += predicted.eq(labels).sum().item(),
    #   early_stopping = EarlyStopping(tolerance=0.3, min_delta=0.3) ### early stopping, as needed,

                # Valdiation model 2nd,
                val_losses = [],
                val_acc = [],
                ,

                def test(epoch):
                    ,

                model.eval(),
                running_loss = 0,
                correct = 0,
                total = 0,
                ,
                with torch.no_grad():
                    ,
                for images, labels in test_loader: ,
                images, labels = images.to(device), labels.to(device),

            ,
            outputs = model.forward(images),
            val_loss = criterion(outputs, labels),
            running_loss += val_loss.item(),
            _, predicted = outputs.max(1),
            total += labels.size(0),
            correct += predicted.eq(labels).sum().item(),
        #   early_stopping = EarlyStopping(tolerance=0.3, min_delta=0.3) ### early stopping, as needed,

    ,
    val_loss = running_loss / len(test_loader),
    acc = 100. * correct / total,
    ,
    val_losses.append(val_loss),
    val_acc.append(acc),
    ,
    print('Val Loss: %.3f | Val Acc: %.3f' % (val_loss, acc)),
    ,
    # early_stopping = EarlyStopping(tolerance =3, min_delta = 0.3),

    ]

    def focal_loss (   ):








    def outputs (  ):