import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch_directml
from CNN_model import ConvNet


device = torch_directml.device()

# torch.manual_seed(111)

# Датасет для обучения
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
train_count = len(train_set)
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

conv_net = ConvNet().to(device)

lr = 0.001
num_epochs = 10
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(conv_net.parameters(), lr=lr)

# loss_list = []
# acc_list = []
print(f'Train images count = {train_count}')
for epoch in range(num_epochs):
    loss = 0
    acc = 0
    for n, (samples, labels) in enumerate(train_loader):
        print(samples.shape)
        samples = samples.to(device)
        labels = labels.to(device)

        conv_net.zero_grad()
        # optimizer.zero_grad()
        output = conv_net(samples)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        # loss_list.append(loss.item())
        # # Track the accuracy
        # total = labels.size(0)
        # _, predicted = torch.max(output.data, 1)
        # # output = output.cpu().detach()
        # correct = (predicted == labels).sum().item()
        # acc = correct / total
        # acc_list.append(acc)

    # print(f"Epoch = {epoch}      Loss = {loss:.3f}      Accuracy = {acc * 100:.2f}%")
    print(f"Epoch = {epoch}      Loss = {loss:.3f}")
    if loss <= 0.01:
        torch.save(conv_net, f'Epoch_{epoch}_Loss_{loss:.3f}.pt')
        print('Model saved.')
