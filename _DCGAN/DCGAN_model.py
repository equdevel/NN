import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


image_shape = (3, 128, 128)
image_dim = int(np.prod(image_shape))
latent_dim = 128  # noise vector size
conv_dim = 128


# Generator Model Class Definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Block 1:input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, conv_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(True),

            # Block 2: (conv_dim * 8) x 4 x 4
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(True),

            # Block 3: (conv_dim * 4) x 8 x 8
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),

            # Block 4: (conv_dim * 2) x 16 x 16
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),

            # Block 5: (conv_dim * 1) x 32 x 32
            nn.ConvTranspose2d(conv_dim * 1, conv_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim // 2),
            nn.ReLU(True),

            # Block 6: (conv_dim * 0.5) x 64 x 64
            nn.ConvTranspose2d(conv_dim // 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (3) x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Discriminator Model Class Definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Block 0: (3) x 128 x 128
            nn.Conv2d(3, conv_dim // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 1: (conv_dim * 0.5) x 64 x 64
            nn.Conv2d(conv_dim // 2, conv_dim * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (conv_dim * 1) x 32 x 32
            nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (conv_dim * 2) x 16 x 16
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (conv_dim * 4) x 8 x 8
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5: (conv_dim * 8) x 4 x 4
            nn.Conv2d(conv_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
            # Output: 1
        )

    def forward(self, input):
        output = self.main(input)
        return output


# custom weights initialization called on generator and discriminator
def weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    if type(m) in (nn.Conv2d, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def show_images(images):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images.detach(), nrow=22).permute(1, 2, 0))
