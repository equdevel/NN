import torch
import matplotlib.pyplot as plt
import torch_directml
from MNIST_GAN_model import Generator


device = torch_directml.device()

# torch.manual_seed(111)

# generator = Generator().to(device)
# generator.load_state_dict(torch.load('Generator_epoch_19_state.pt'))
generator = torch.load('Generator_epoch_25_LossG_0.855.pt', map_location=device)

# Тест генератора
latent_space_samples = torch.randn(16, 100).to(device)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.cpu().detach()
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
plt.show()
