import torch
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 1. Generator (G) - Generates fake graph samples from random noise
class Generator(nn.Module):
    def __init__(self, noise_dim, graph_feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            # Output represents the spatial features of the generated graph
            nn.Linear(256, graph_feature_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# 2. Discriminator (D) - Classifies samples as Real (1) or Fake (0)
class Discriminator(nn.Module):
    def __init__(self, graph_feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(graph_feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_sample):
        return self.model(graph_sample)


# 3. Training Logic (Algorithm 1 / Fig 1 Workflow)
def train_sg_gan(real_data_loader, noise_dim, feature_dim):
    # Initialize networks
    netG = Generator(noise_dim, feature_dim)
    netD = Discriminator(feature_dim)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

    for epoch in range(100):
        for i, real_samples in enumerate(real_data_loader):
            batch_size = real_samples.size(0)

            # --- Update Discriminator ---
            netD.zero_grad()
            # Real Graph Samples
            labels_real = torch.ones(batch_size, 1)
            output_real = netD(real_samples)
            loss_D_real = criterion(output_real, labels_real)

            # Generated Fake Graph Samples
            noise = torch.randn(batch_size, noise_dim)
            fake_samples = netG(noise)
            labels_fake = torch.zeros(batch_size, 1)
            output_fake = netD(fake_samples.detach())
            loss_D_fake = criterion(output_fake, labels_fake)

            # Backpropagation for Discriminator Loss (Fig 1)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizerD.step()

            # --- Update Generator ---
            netG.zero_grad()
            # G wants D to classify fake samples as Real (1)
            output_g = netD(fake_samples)
            loss_G = criterion(output_g, labels_real)

            # Backpropagation for Generator Loss (Fig 1)
            loss_G.backward()
            optimizerG.step()

    return netG, netD





def test_and_visualize(generator, noise_dim, real_features):
    """
    Tests the SG-GAN by generating a sample and comparing it to real data.
    """
    # 1. Generate Fake Graph Sample
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, noise_dim)
        # Assuming the generator outputs a sequence of (x, y) coordinates for nodes
        generated_sample = generator(noise).cpu().numpy().reshape(-1, 2)

    # 2. Visualize the Spatial Graph
    plt.figure(figsize=(8, 6))
    G = nx.Graph()

    # Add nodes and edges (simplified connectivity for testing)
    for i, (x, y) in enumerate(generated_sample):
        G.add_node(i, pos=(x, y))
        if i > 0:
            G.add_edge(i - 1, i)  # Simple path connectivity

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.title("Generated Spatial Road Network Sample")
    plt.show()

    # 3. Statistical Testing: KL Divergence (Eq. 11)
    # Let's compare the distribution of edge lengths
    def get_lengths(nodes):
        return np.sqrt(np.sum(np.diff(nodes, axis=0) ** 2, axis=1))

    real_dist, _ = np.histogram(get_lengths(real_features), bins=10, density=True)
    fake_dist, _ = np.histogram(get_lengths(generated_sample), bins=10, density=True)

    # Adding a small constant to avoid division by zero (Laplace smoothing)
    real_dist += 1e-10
    fake_dist += 1e-10

    kl_div = entropy(real_dist, fake_dist)
    print(f"KL Divergence Score: {kl_div:.4f}")

    if kl_div < 0.5:
        print("Test Passed: The generated network distribution is statistically similar to the real one.")
    else:
        print("Test Failed: High divergence. Generator needs more training.")

# Example usage:
test_and_visualize(netG, noise_dim=10, real_features=my_training_data)