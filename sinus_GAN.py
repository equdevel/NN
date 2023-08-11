import torch
from torch import nn
import math
import matplotlib.pyplot as plt
from sinus_GAN_model import Discriminator, Generator
import torch_directml


device = torch_directml.device()

torch.manual_seed(111)

# Create neural nets
discriminator = Discriminator()  #.to(device)
generator = Generator()  #.to(device)

# Dataset for learning
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))  #.to(device)
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)  #.to(device)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

# train_data = train_data.cpu()
# plt.plot(train_data[:, 0], train_data[:, 1], ".")
# plt.show()

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Learning
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

fixed_latent_space_samples = torch.randn((batch_size, 2))

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # real_samples = real_samples.to(device)
        # Данные для обучения дискриминатора
        real_samples_labels = torch.ones((batch_size, 1))  #.to(device)
        latent_space_samples = torch.randn((batch_size, 2))  #.to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))  #.to(device)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Обучение дискриминатора
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Данные для обучения генератора
        latent_space_samples = torch.randn((batch_size, 2))  #.to(device)

        # Обучение генератора
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Выводим значения функций потерь
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            # generated_samples = generated_samples.detach()
            generated_samples = generator(fixed_latent_space_samples).detach()
            plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
            # plt.scatter(generated_samples[:, 0], generated_samples[:, 1])
            plt.show()

# Test
latent_space_samples = torch.randn(100, 2)  #.to(device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()  #.cpu()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()
