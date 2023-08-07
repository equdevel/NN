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
        self.img_labels = np.array(data_train_dict[b'coarse_labels'])
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
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(8 * 8 * 64, 1000),
            nn.Linear(1000, 20)
        )

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.model(x)
