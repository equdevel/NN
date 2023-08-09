import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch_directml
from CNN_CIFAR100_model import ConvNet, CIFAR100_dataset


device = torch_directml.device()
# torch.manual_seed(111)

# train_set = tuple(zip(data_train, label_train))
transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize((0.5,), (0.5,))])
train_set = CIFAR100_dataset('CIFAR-100/train', transform=transform)
train_count = len(train_set)
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

conv_net = ConvNet().to(device)
# conv_net = torch.load('CIFAR100_100-class__Epoch_0_Loss_0.001.pt', map_location=device)  # conv_net.train()

lr = 0.0001
num_epochs = 30
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

        # conv_net.zero_grad()
        optimizer.zero_grad()
        output = conv_net(samples)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch = {epoch}      Loss = {loss:.4f}")
    if loss <= 0.001:
        torch.save(conv_net, f'CIFAR100_100-class__Epoch_{epoch}_Loss_{loss:.4f}.pt')
        print('Model saved.')
torch.save(conv_net, f'CIFAR100_100-class__last_epoch_Loss_{loss:.4f}.pt')
