import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ebflow.datasets.datatransforms import ToTensorNoNorm

def load_data(data_aug=False, batch_size=100, size=28, **kwargs):
    transform_list = []

    if data_aug:
        transform_list.append(transforms.Pad(1, padding_mode='reflect'))
        transform_list.append(transforms.RandomCrop(28))

    transform_list.append(transforms.Resize(size))
    transform_list.append(ToTensorNoNorm())

    transform = transforms.Compose(transform_list)
    trainvalset = datasets.KMNIST('../data',
                                 train=True,
                                 download=True,
                                 transform=transform)
    testset = datasets.KMNIST('../data', train=False, transform=transform)

    trainset = torch.utils.data.Subset(trainvalset, range(0, 50_000))
    valset = torch.utils.data.Subset(trainvalset, range(50_000, 60_000))

    train_loader = DataLoader(trainset, batch_size, **kwargs)
    val_loader = DataLoader(valset, batch_size//2, **kwargs)
    test_loader = DataLoader(testset, batch_size//2, **kwargs)

    return train_loader, val_loader, test_loader