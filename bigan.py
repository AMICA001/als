import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_shape):
        super(Generator, self).__init__()
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, np.prod(output_shape)),
            nn.Tanh()
        )

    def forward(self, z, c):
        # Concatenate latent vector z and condition vector c
        input_vec = torch.cat((z, c), -1)
        img = self.model(input_vec)
        img = img.view(img.size(0), *self.output_shape)
        return img

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, latent_dim)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        z = self.model(img_flat)
        return z

class Discriminator(nn.Module):
    def __init__(self, input_shape, latent_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape) + latent_dim + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, z, c):
        img_flat = img.view(img.size(0), -1)
        # Concatenate image, latent vector, and condition vector
        input_vec = torch.cat((img_flat, z, c), -1)
        validity = self.model(input_vec)
        return validity

def train_bigan(residual_surfaces, market_conditions, latent_dim=4, n_epochs=200, lr=0.0002, b1=0.5, b2=0.999):
    """
    Trains the BiGAN model.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reshape and convert data to tensors
    num_surfaces = residual_surfaces.shape[0]
    grid_size = int(np.sqrt(residual_surfaces.shape[1]))
    residual_surfaces_3d = residual_surfaces.values.reshape(num_surfaces, grid_size, grid_size)
    residual_surfaces = torch.from_numpy(residual_surfaces_3d).float().to(device)
    market_conditions = torch.from_numpy(market_conditions.values).float().to(device)

    input_shape = (grid_size, grid_size)
    condition_dim = market_conditions.shape[1]

    # Initialize generator, encoder, and discriminator
    generator = Generator(latent_dim, condition_dim, input_shape).to(device)
    encoder = Encoder(input_shape, latent_dim).to(device)
    discriminator = Discriminator(input_shape, latent_dim, condition_dim).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_E = optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Adversarial loss
    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(n_epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample real data
        real_imgs = residual_surfaces
        real_conditions = market_conditions

        # Generate fake data
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z, real_conditions).detach()

        # Encode real images
        real_z = encoder(real_imgs)

        # Discriminator loss for real and fake data
        real_loss = adversarial_loss(discriminator(real_imgs, real_z, real_conditions), torch.ones(real_imgs.size(0), 1).to(device))
        fake_loss = adversarial_loss(discriminator(fake_imgs, z, real_conditions), torch.zeros(fake_imgs.size(0), 1).to(device))
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator and Encoder
        # -----------------
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        # Generate fake data
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z, real_conditions)

        # Encode generated images
        gen_z = encoder(gen_imgs)

        # Loss for generator and encoder
        g_loss = adversarial_loss(discriminator(gen_imgs, z, real_conditions), torch.ones(gen_imgs.size(0), 1).to(device))
        e_loss = adversarial_loss(discriminator(real_imgs, gen_z, real_conditions), torch.zeros(real_imgs.size(0), 1).to(device))
        ge_loss = g_loss + e_loss

        ge_loss.backward()
        optimizer_G.step()
        optimizer_E.step()

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f] [E loss: %f]"
            % (epoch, n_epochs, d_loss.item(), g_loss.item(), e_loss.item())
        )

    return generator, encoder
