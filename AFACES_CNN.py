import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch_directml
from PIL import Image
import torchvision.transforms as transforms
from AFACES_CNN_model import AnimalsImageDataset


device = torch_directml.device()
# torch.manual_seed(111)

PATH = 'ANIMAL_FACES/train'
# !!!!!! transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = AnimalsImageDataset(f'{PATH}/labels.csv', PATH)
train_count = len(train_set)
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

label_name = {0: 'cat', 1: 'dog', 2: 'wild'}
# print(train_count)
# transform = transforms.ToPILImage()
# print(train_set[0][0].shape)
samples, labels = next(iter(train_loader))
# plt.imshow(transform(samples[0]))
plt.imshow(samples[0].permute(1, 2, 0))
plt.title(label_name[labels[0].item()])
plt.show()
