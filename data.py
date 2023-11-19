import random
import torch
import torchvision
from glob import glob
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import make_grid

from labels import mapping


def create_mapper():
    mapper = torch.zeros(40, dtype=torch.long)
    for k, v in mapping.items():
        mapper[k] = v
    return mapper

mapper = create_mapper()

label_transform = transform.Compose([
    transform.ToTensor(),
    transform.Lambda(lambda x: x * 255),
    transform.Lambda(lambda x: mapper[x.long()].long())
])

val_test_transforms = transform.Compose([
    transform.ToTensor()
])

class CityscapesDataset(Dataset):
    
    def __init__(self, images_path, labels_path, test=False):
        self.images_path = images_path
        self.labels_path = labels_path
        self.test = test

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert("RGB")
        label = Image.open(self.labels_path[idx]).convert("P")

        if self.test:
            img = val_test_transforms(img)
            label = label_transform(label)
        else:
            img, label = self.apply_transforms(img, label)

        return img, label

    def apply_transforms(self, img, label):
        
        # Apply geometric transformations
        if random.random() > 0.5:
            img = transform.functional.hflip(img)
            label = transform.functional.hflip(label)
        
        if random.random() > 0.5:
            img = transform.functional.vflip(img)
            label = transform.functional.vflip(label)

        # Add other transformations here as needed, ensuring they are applied to both img and label

        # Apply ColorJitter only to the image
        img = transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)

        # Finalize with ToTensor and Resize
        img = transform.ToTensor()(img)
        img = transform.Resize((1024, 2048))(img)
        label = label_transform(label)

        return img, label
        