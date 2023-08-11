import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch_directml
from MNIST_GAN_model import Discriminator, Generator


device = torch_directml.device()

# torch.manual_seed(111)

# Датасет для обучения
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
# batch_count = len(train_loader)

# real_samples, mnist_labels = next(iter(train_loader))
# for i in range(16):
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

discriminator = Discriminator().to(device)
generator = Generator().to(device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

fixed_latent_space_samples = torch.randn((batch_size, 100)).to(device)

# print(len(train_set), len(train_loader))
for epoch in range(num_epochs):
    loss_discriminator, loss_generator = 0, 0
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # Данные для тренировки дискриминатора
        real_samples = real_samples.to(device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device)
        latent_space_samples = torch.randn((batch_size, 100)).to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Обучение дискриминатора
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Данные для обучения генератора
        latent_space_samples = torch.randn((batch_size, 100)).to(device)

        # Обучение генератора
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
    # Показываем loss
    print(f"Epoch = {epoch}      Loss_D = {loss_discriminator:.3f}      Loss_G = {loss_generator:.3f}")
    generated_samples = generator(fixed_latent_space_samples)
    generated_samples = generated_samples.cpu().detach()
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    # if (epoch+1) % 10 == 0:
    if loss_generator <= 1:
        # torch.save(generator.state_dict(), f'Generator_epoch_{epoch}_state.pt')
        # print('Generator model state_dict saved.')
        torch.save(generator, f'Generator_epoch_{epoch}_LossG_{loss_generator:.3f}.pt')
        print('Generator model saved.')
