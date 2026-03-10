import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy


# --- 1. ROBUST DATA EXTRACTION (Dwarka Mor) ---
def get_real_world_data():
    """Fetches real road network data using precise coordinates."""
    # Dwarka Mor, New Delhi Coordinates
    lat, lon = 28.6186, 77.0315

    print(f"Fetching road network for Dwarka Mor ({lat}, {lon})...")

    try:
        # Use a point-based query to avoid Geocoding errors
        graph = ox.graph_from_point((lat, lon), dist=1000, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values

        # Min-Max Normalization to [-1, 1] as per Eq. 3
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # Scaling calculation
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]

        print(f"Successfully extracted {len(norm_coords)} road nodes.")
        return torch.FloatTensor(norm_coords), (x_min, x_max, y_min, y_max)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Falling back to synthetic data for demonstration...")
        return torch.randn(100, 2), (0, 1, 0, 1)


# --- 2. SG-GAN MODELS (Fig 1) ---
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # Normalizes output to [-1, 1] range
        )

    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Real (1) vs Fake (0)
        )

    def forward(self, x): return self.main(x)


# --- 3. TRAINING & EVALUATION (Algorithm 1) ---
def train_and_test():
    # Model Configuration
    noise_dim = 10
    sample_size = 10
    feature_dim = sample_size * 2  # X, Y pairs
    epochs = 300

    # Load Real Data from Dwarka Mor
    real_data, bounds = get_real_world_data()

    # Initialize networks
    netG = Generator(noise_dim, feature_dim)
    netD = Discriminator(feature_dim)
    optG = optim.Adam(netG.parameters(), lr=0.0002)
    optD = optim.Adam(netD.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    print("Training SG-GAN...")
    for epoch in range(epochs):
        # 1. Prepare Real Samples
        idx = np.random.randint(0, len(real_data), sample_size)
        real_batch = real_data[idx].view(1, -1)

        # 2. Update Discriminator (Algorithm 1, Lines 8-12)
        optD.zero_grad()
        out_real = netD(real_batch)
        loss_real = criterion(out_real, torch.ones(1, 1))

        noise = torch.randn(1, noise_dim)
        fake_batch = netG(noise)
        out_fake = netD(fake_batch.detach())
        loss_fake = criterion(out_fake, torch.zeros(1, 1))

        d_loss = loss_real + loss_fake
        d_loss.backward()
        optD.step()

        # 3. Update Generator (Fig 1 Backpropagation)
        optG.zero_grad()
        out_g = netD(fake_batch)
        g_loss = criterion(out_g, torch.ones(1, 1))
        g_loss.backward()
        optG.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D-Loss: {d_loss.item():.4f} | G-Loss: {g_loss.item():.4f}")

    # --- 4. FINAL VISUALIZATION & KL CHECK ---
    final_fake = netG(torch.randn(1, noise_dim)).detach().numpy().reshape(-1, 2)

    # KL Divergence check (Eq. 11)
    p, _ = np.histogram(real_batch.numpy(), bins=10, density=True)
    q, _ = np.histogram(final_fake, bins=10, density=True)
    kl_div = entropy(p + 1e-10, q + 1e-10)
    print(f"Final KL Divergence: {kl_div:.4f}")

    # Plot Generated Points
    plt.figure(figsize=(8, 6))
    plt.scatter(final_fake[:, 0], final_fake[:, 1], c='red', label='Generated Road Nodes')
    plt.title("SG-GAN Generated Road Intersections (Dwarka Mor Simulation)")
    plt.xlabel("Normalized Longitude")
    plt.ylabel("Normalized Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_and_test()