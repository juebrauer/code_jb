import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os


# --- Generator ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# --- CelebA Dataset ---
# Du musst den Ordner vorher herunterladen, z. B.:
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# und in ./data/celeba/img_align_celeba/ ablegen



transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = datasets.CelebA(root="celeba", split="train", transform=transform, download=True)



#dataset = datasets.ImageFolder(
#    root="celeba",
#    transform=transform
#)

loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)


# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

#for z_dim in range(10,1000,10):
z_dim = 100

gen = Generator(z_dim).to(device)
disc = Discriminator().to(device)

opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion = nn.BCELoss()

os.makedirs("generated_samples", exist_ok=True)


# --- Training ---
epochs = 100
fixed_noise = torch.randn(16, z_dim, 1, 1).to(device)


for epoch in range(1, epochs + 1):
    for real, _ in loader:
        real = real.to(device)
        batch_size = real.size(0)

        ### Train discriminator ###
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real)
        loss_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach())
        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_d = (loss_real + loss_fake) / 2

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        ### Train generator ###
        output = disc(fake)
        loss_g = criterion(output, torch.ones_like(output))

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print(f"Epoch {epoch}/{epochs}  D Loss: {loss_d.item():.4f}  G Loss: {loss_g.item():.4f}")

    # --- Bilder generieren + speichern ---
    gen.eval()
    with torch.no_grad():
        samples = gen(fixed_noise).cpu()
        samples = (samples + 1) / 2  # zurück von Tanh zu [0,1]

    grid = utils.make_grid(samples, nrow=4)

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")

    fname = f"generated_samples/generated_samples_{epoch:04d}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    gen.train()
