import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# --- VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 400)
        self.fc_dec2 = nn.Linear(400, 28*28)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        return torch.sigmoid(self.fc_dec2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# --- Loss ---
def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


# --- Daten ---
transform = transforms.ToTensor()
dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VAE(latent_dim=20).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Output-Ordner
output_dir = "generated_samples"
os.makedirs(output_dir, exist_ok=True)


# --- Training über 500 Epochen ---
for epoch in range(1, 501):
    total_loss = 0
    model.train()
    for x, _ in loader:
        x = x.to(device).view(-1, 28*28)
        optim.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optim.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}/500  Loss: {total_loss/len(loader):.2f}")

    # --- 16 Beispiele generieren, visualisieren und speichern ---
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, 20).to(device)
        samples = model.decode(z).cpu().view(16, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for ax, img in zip(axes.flat, samples):
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    filename = os.path.join(output_dir, f"generated_samples_{epoch:04d}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
