import os

import numpy as np
import pandas as pd

import torch
import torchvision
from glob import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import make_grid

from nets import SimpleSegmentationNet
from data import CityscapesDataset
from labels import labels
from unet import UNet


def train(
    model,
    optimizer,
    criterion,
    epochs,
    train_loader,
    val_loader,
    device,
    checkpoint_interval,
    start_epoch=0
):
    train_loss = []
    val_loss = []
    val_accuracy = []


    checkpoint_dir = "checkpoints/unet_trans"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i in range(start_epoch, start_epoch + epochs):
        
        model.train()
        trainloss = 0
        
        for data in tqdm(train_loader, desc=f'Epoch {i+1}/{epochs} [Training]'):
            # Training
            img, label = data[0].to(device), data[1].to(device)
            label = label.squeeze(1).to(dtype=torch.long)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            trainloss += loss.item()

        train_loss.append(trainloss / len(train_loader))    
        
        model.eval()

        with torch.no_grad():
            valloss = 0
            total_correct = 0
            total_pixels = 0
            
            for data in val_loader:
                # Validation
                img, label = data[0].to(device), data[1].to(device)
                label = label.squeeze(1).to(dtype=torch.long)
                output = model(img)
                loss = criterion(output, label)
                valloss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_pixels += label.nelement()

        val_loss.append(valloss / len(val_loader))
        val_accuracy.append(total_correct / total_pixels)
        
        print("Epoch: {} , Train Loss: {} , Valid Loss: {} , Valid Acc: {:.2f}%".format(i, train_loss[-1], val_loss[-1], 100 * val_accuracy[-1]))

        if i%checkpoint_interval == 0:
            # Checkpointing
            checkpoint = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{i}.pth'))

    return train_loss, val_loss, val_accuracy



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_seg_path = sorted(glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelIds.png'))
    train_img_path = sorted(glob('/cluster/projects/vc/data/ad/open/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png'))

    val_seg_path = sorted(glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/val/*/*labelIds.png'))
    val_img_path = sorted(glob('/cluster/projects/vc/data/ad/open/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/*/*.png'))

    traindata = CityscapesDataset(train_img_path, train_seg_path)
    valdata = CityscapesDataset(val_img_path, val_seg_path, test=True)

    batch_size = 2
    train_loader = DataLoader(traindata, batch_size)
    val_loader = DataLoader(valdata, batch_size)

    num_classes = len(labels)
    model = UNet(3, num_classes)
    model.to(device)

    lr = 0.01
    epochs = 21
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0

    train_loss, val_loss, val_accuracy = train(
        model = model,
        optimizer=optimizer,
        criterion = criterion,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_interval=5,
        start_epoch=start_epoch
    )

    print("finished")

if __name__ == "__main__":
    main()
