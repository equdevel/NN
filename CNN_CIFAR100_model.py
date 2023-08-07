import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR100_dataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        data_train_dict = unpickle(img_dir)  # 'CIFAR-100/train'
        self.img_labels = np.array(data_train_dict[b'fine_labels'])
        self.img_samples = data_train_dict[b'data'].reshape(-1, 3, 32, 32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_samples[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=2, stride=1),
            nn.Dropout(p=0.1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, stride=1),
            nn.Dropout(p=0.1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2, stride=1),
            nn.Dropout(p=0.1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256*5*5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.model(x)
