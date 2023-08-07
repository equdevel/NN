import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch_directml
import torchvision
import torchvision.transforms as transforms
import numpy as np


device = torch_directml.device()

# torch.manual_seed(111)

# Датасет для теста
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = torchvision.datasets.MNIST(root=".", train=False, transform=transform)
test_count = len(test_set)
batch_size = 16
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

conv_net = torch.load('Epoch_9_Loss_0.006.pt', map_location=device)

samples, labels = next(iter(test_loader))
output = conv_net(samples.to(device))
output = output.cpu().detach()
L = np.array(labels).reshape((4, 4))
P = np.array(torch.max(output.data, 1)[1].reshape((4, 4)))
print('Labels =')
print(L)
print('Predictions =')
print(P)
for i in range(batch_size):
    # print(labels[i].item(), '  =  ', torch.max(output[i].data, 0)[1].item())
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

# Test the model
conv_net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        output = conv_net(samples)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Test accuracy of the model on {test_count} test images: {(correct / total) * 100:.2f} %')
