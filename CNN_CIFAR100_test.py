import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch_directml
import torchvision
import torchvision.transforms as transforms
import numpy as np
from CNN_CIFAR100_model import CIFAR100_dataset, unpickle


device = torch_directml.device()

# torch.manual_seed(111)

# Датасет для теста
metadata = unpickle('CIFAR-100/meta')
superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
class_dict = dict(list(enumerate(metadata[b'fine_label_names'])))

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
test_set = CIFAR100_dataset('CIFAR-100/test', transform=transform)
test_count = len(test_set)
batch_size = 9
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

conv_net = torch.load('CIFAR100_Epoch_48_Loss_0.016.pt', map_location=device)

samples, labels = next(iter(test_loader))
output = conv_net(samples.to(device))
output = output.cpu().detach()
L = np.array(labels).reshape((3, 3))
P = np.array(torch.max(output.data, 1)[1].reshape((3, 3)))
print('Labels =')
print(L)
print('Predictions =')
print(P)
for i in range(batch_size):
    plt.subplot(3, 3, i + 1)
    plt.imshow(samples[i].permute(2, 0, 1))
    label = torch.max(output.data, 1)[1][i].item()
    plt.title(class_dict[label])
    plt.xticks([])
    plt.yticks([])
plt.show()

# Test the model
print(f'Test images count = {test_count}\n')
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
        correct += sum(predicted == labels).item()
    print(f'Test accuracy of the model on {test_count} test images: {(correct / total) * 100:.2f} %')
