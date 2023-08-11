import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import torch_directml
from PIL import Image
from torchvision.utils import make_grid
from DCGAN_model import Generator, Discriminator, latent_dim, weights_init


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# torch.manual_seed(1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch_directml.device()
batch_size = 128

train_transform = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])
                                      # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = datasets.ImageFolder(root='../../afhq/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# im = train_dataset[12][0].permute(1, 2, 0)
# plt.imshow(im)
# plt.show()

generator = Generator().to(device)
generator.apply(weights_init)
# print(generator)
# summary(generator, (100, 1, 1))

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
# print(discriminator)
# summary(discriminator, (3,64,64))

# Continue train
generator = torch.load('training_weights/generator_epoch_7_Gloss_1.867.pth', map_location=device)
discriminator = torch.load('training_weights/discriminator_epoch_7_Dloss_1.371.pth', map_location=device)

generator_loss = nn.BCELoss()
discriminator_loss = nn.BCELoss()

fixed_noise = torch.randn(128, latent_dim, 1, 1, device=device)
real_label = 1
fake_label = 0

learning_rate = 0.0001
G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

num_epochs = 50
D_loss_plot, G_loss_plot = [], []
for epoch in range(1, num_epochs+1): 

    D_loss_list, G_loss_list = [], []
   
    for index, (real_images, _) in enumerate(train_loader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
      
        real_target = Variable(torch.ones(real_images.size(0)).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0)).to(device))

        real_target = real_target.unsqueeze(1)
        fake_target = fake_target.unsqueeze(1)

        D_real_loss = discriminator_loss(discriminator(real_images), real_target)
        # print(discriminator(real_images))
        D_real_loss.backward()
    
        noise_vector = torch.randn(real_images.size(0), latent_dim, 1, 1, device=device)  
        noise_vector = noise_vector.to(device)
    
        generated_image = generator(noise_vector)
        output = discriminator(generated_image.detach())
        D_fake_loss = discriminator_loss(output,  fake_target)

        # train with fake
        D_fake_loss.backward()
      
        D_total_loss = D_real_loss + D_fake_loss
        D_loss_list.append(D_total_loss)
      
        # D_total_loss.backward()
        D_optimizer.step()

        # Train generator with real labels
        G_optimizer.zero_grad()
        G_loss = generator_loss(discriminator(generated_image), real_target)
        G_loss_list.append(G_loss)

        G_loss.backward()
        G_optimizer.step()

    D_loss = torch.mean(torch.FloatTensor(D_loss_list)).item()
    G_loss = torch.mean(torch.FloatTensor(G_loss_list)).item()
    D_loss_plot.append(D_loss)
    G_loss_plot.append(G_loss)
    print('Epoch [%d/%d]: D_loss = %.3f, G_loss = %.3f' % (epoch, num_epochs, D_loss, G_loss))

    # if G_loss <= 2.5:
    torch.save(generator, f'training_weights/generator_epoch_{epoch}_Gloss_{G_loss:.3f}.pth')
    torch.save(discriminator, f'training_weights/discriminator_epoch_{epoch}_Dloss_{D_loss:.3f}.pth')
    print('Model saved.')

    # im = generated_image[0].cpu().detach().permute(1, 2, 0)
    # im = torch.sigmoid(im)
    # im = np.array(im)
    # im = (im * 255).astype(np.uint8)
    # im = (im * 255).to(torch.uint8)  # is equivalent to .byte()
    # plt.imshow(im)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # im = Image.fromarray(im)
    # im.save('generated_images/0.png')

    im = generated_image[:20].cpu().detach()  # .permute(0, 2, 3, 1)
    # im = (im * 255).to(torch.uint8)
    save_image(im, f'generated_images/sample_{epoch}.png', nrow=5, normalize=True, value_range=(0, 1))

    im = generated_image[:4].cpu().detach()
    im = make_grid(im, nrow=2, normalize=True, value_range=(0, 1)).permute(1, 2, 0)
    plt.imshow(im)
    plt.show()