import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch_directml
from PIL import Image
import torchvision.transforms as transforms
from AFACES_CNN_model import AnimalsImageDataset, ConvNet


device = torch_directml.device()
# torch.manual_seed(111)

PATH = 'ANIMAL_FACES/train'
# transform = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
transform = transforms.Compose([transforms.ToTensor()])
train_set = AnimalsImageDataset(f'{PATH}/labels.csv', PATH, transform=transform)
train_count = len(train_set)
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
label_name = {0: 'cat', 1: 'dog', 2: 'wild'}

conv_net = ConvNet().to(device)
lr = 0.001
num_epochs = 50
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(conv_net.parameters(), lr=lr)

print(f'Train images count = {train_count}')
for epoch in range(num_epochs):
    loss = 0
    acc = 0
    for n, (samples, labels) in enumerate(train_loader):
        # print(samples.shape)
        samples = samples.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = conv_net(samples)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch = {epoch}      Loss = {loss:.4f}")
    if loss <= 0.01:
        torch.save(conv_net, f'AFACES__Epoch_{epoch}_Loss_{loss:.4f}.pt')
        print('Model saved.')
torch.save(conv_net, f'AFACES__last_epoch_Loss_{loss:.4f}.pt')

# samples, labels = next(iter(train_loader))
# plt.imshow(samples[0].permute(1, 2, 0))
# plt.title(label_name[labels[0].item()])
# plt.xticks([])
# plt.yticks([])
# plt.show()
