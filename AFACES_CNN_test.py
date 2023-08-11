import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from AFACES_CNN_model import AnimalsImageDataset, ConvNet
import torch_directml


device = torch_directml.device()
# device = torch.device('cpu')

PATH = 'ANIMAL_FACES/val'
transform = transforms.Compose([transforms.ToTensor()])
test_set = AnimalsImageDataset(f'{PATH}/labels.csv', PATH, transform=transform)
test_count = len(test_set)
batch_size = 100
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
label_name = {0: 'cat', 1: 'dog', 2: 'wild'}

# Test the model
# conv_net = ConvNet()
# conv_net.load_state_dict(torch.load('AFACES__Epoch_18_Loss_0.0000.pth', map_location=device))
# conv_net.to(device)
conv_net = torch.load('AFACES__Epoch_18_Loss_0.0000.pt', map_location=torch.device('cpu'))
conv_net.to(device)

print(f'Test images count = {test_count}\n')
conv_net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for n, (samples, labels) in enumerate(test_loader):
        samples = samples.to(device)
        labels = labels.to(device)
        output = conv_net(samples)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels).item()
        # Show some predicted image
        pred = predicted[0].item()
        sample = samples[0].permute(1, 2, 0)
        plt.imshow(sample.cpu())
        plt.title(label_name[pred])
        plt.xticks([])
        plt.yticks([])
        plt.show()
    print(f'Test accuracy of the model on {test_count} test images: {(correct / total) * 100:.2f} %')
