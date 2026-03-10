import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy


# --- 1. DATA EXTRACTION (Dwarka Mor) ---
def get_real_world_data():
    lat, lon = 28.6186, 77.0315
    try:
        graph = ox.graph_from_point((lat, lon), dist=1000, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        return torch.FloatTensor(norm_coords), (x_min, x_max, y_min, y_max)
    except Exception as e:
        return torch.randn(100, 2), (0, 1, 0, 1)


# --- 2. SG-GAN MODELS (Fig. 1) ---
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Tanh()
        )

    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.main(x)


# --- 3. WEIGHTED COST CALCULATION (Section IV) ---
def calculate_edge_weight(dist_km, beta=0.5):
    """Calculates weight O based on Eq. 11"""
    # Constants from Eq. 6 (Simplified for demonstration)
    v = 40 / 3.6  # 40 km/h to m/s
    eta = 0.9  # Efficiency
    m = 1500  # Mass
    g = 9.81

    # Energy E_ci (Eq. 6) simplified
    energy = (1 / eta) * (0.5 * 1.225 * 0.3 * 2.5 * (v ** 2) + 0.01 * m * g) * (dist_km * 1000 / v)

    # Travel Time T_ci (Eq. 7) simplified (assuming zero queueing for weight init)
    travel_time = (dist_km * 1000) / v

    # Normalization factors (Eq. 11)
    E_norm, T_norm = 100000, 3600

    # Final Weight O (Objective Function)
    weight = beta * (energy / E_norm) + (1 - beta) * (travel_time / T_norm)
    return weight


# --- 4. MAIN EXECUTION ---
def run_weighted_simulation():
    noise_dim, sample_size = 10, 15
    feature_dim = sample_size * 2

    real_data, _ = get_real_world_data()
    netG, netD = Generator(noise_dim, feature_dim), Discriminator(feature_dim)
    optG, optD = optim.Adam(netG.parameters(), lr=0.0005), optim.Adam(netD.parameters(), lr=0.0005)

    print("Training SG-GAN and Building Weighted Graph...")
    for epoch in range(501):
        idx = np.random.randint(0, len(real_data), sample_size)
        real_batch = real_data[idx].view(1, -1)

        # Train Discriminator
        optD.zero_grad()
        loss_d = nn.BCELoss()(netD(real_batch), torch.ones(1, 1)) + \
                 nn.BCELoss()(netD(netG(torch.randn(1, noise_dim)).detach()), torch.zeros(1, 1))
        loss_d.backward();
        optD.step()

        # Train Generator
        optG.zero_grad()
        loss_g = nn.BCELoss()(netD(netG(torch.randn(1, noise_dim))), torch.ones(1, 1))
        loss_g.backward();
        optG.step()

    # Generate Nodes and Create Weighted Graph
    final_nodes = netG(torch.randn(1, noise_dim)).detach().numpy().reshape(-1, 2)
    G = nx.Graph()

    for i, pos in enumerate(final_nodes):
        G.add_node(i, pos=pos)

    # Establish Edges and Weights
    for i in range(len(final_nodes)):
        for j in range(i + 1, len(final_nodes)):
            dist = np.linalg.norm(final_nodes[i] - final_nodes[j])
            if dist < 0.4:  # Proximity threshold for road connectivity
                w = calculate_edge_weight(dist)
                G.add_edge(i, j, weight=w)

    # Visualization
    plt.figure(figsize=(10, 7))
    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]  # Scale for visibility

    nx.draw(G, pos, with_labels=True, node_color='orange',
            width=weights, edge_color='blue', node_size=300)
    plt.title("Dwarka Mor: SG-GAN Generated Weighted Road Graph\n(Edge thickness = Objective Cost O)")
    plt.show()


if __name__ == "__main__":
    run_weighted_simulation()