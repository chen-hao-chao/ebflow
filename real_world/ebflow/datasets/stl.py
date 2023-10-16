import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from ebflow.datasets.datatransforms import ToTensorNoNorm
import math
import numpy as np

def load_data(data_aug=True, **kwargs):
    assert data_aug == True
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(int(math.ceil(96 * 0.04)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.CenterCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(64),
        ToTensorNoNorm()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        ToTensorNoNorm()
    ])

    data_train = torchvision.datasets.STL10('./data', split="train", transform=train_transform,
                                              target_transform=None, download=True)

    train = torch.utils.data.Subset(data_train, torch.arange(0, 5000))

    data_val = torchvision.datasets.STL10('./data', split="train",
                                              transform=test_transform,
                                              target_transform=None,
                                              download=False)

    val = torch.utils.data.Subset(data_val, torch.arange(0, 5000))

    test = torchvision.datasets.STL10('./data', split="train",
                                          transform=test_transform,
                                          target_transform=None,
                                          download=True)

    train_loader = data_utils.DataLoader(train, 
                                         shuffle=True, **kwargs)

    val_loader = data_utils.DataLoader(val, 
                                       shuffle=False, **kwargs)

    test_loader = data_utils.DataLoader(test,
                                        shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
