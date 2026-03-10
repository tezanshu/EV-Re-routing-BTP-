import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

N_NODES = 29
N_CS = 7
N_EV = 50
NOISE_DIM = 10

def get_spatial_features():
    lat, lon = 28.6186, 77.0315
    try:
        graph = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        return torch.FloatTensor(norm_coords)
    except:
        return torch.randn(100, 2)

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def generate_dwarka_mor_network():
    real_data = get_spatial_features()

    netG = Generator(NOISE_DIM, N_NODES * 2)
    netD = Discriminator(N_NODES * 2)
    optimizer_g = optim.Adam(netG.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(netD.parameters(), lr=0.0002)

    print("Synthesizing Road Network via SG-GAN...")
    for epoch in range(10000):
        idx = np.random.randint(0, len(real_data), N_NODES)
        real_batch = real_data[idx].view(1, -1)

        optimizer_d.zero_grad()
        noise = torch.randn(1, NOISE_DIM)
        fake_batch = netG(noise)
        d_loss = nn.BCELoss()(netD(real_batch), torch.ones(1, 1)) + \
                 nn.BCELoss()(netD(fake_batch.detach()), torch.zeros(1, 1))
        d_loss.backward(); optimizer_d.step()

        optimizer_g.zero_grad()
        g_loss = nn.BCELoss()(netD(fake_batch), torch.ones(1, 1))
        g_loss.backward(); optimizer_g.step()

    nodes_coords = netG(torch.randn(1, NOISE_DIM)).detach().numpy().reshape(-1, 2)

    for _ in range(50):
        for i in range(N_NODES):
            for j in range(i+1, N_NODES):
                dist = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
                if dist < 0.15:
                    direction = (nodes_coords[i] - nodes_coords[j]) / (dist + 1e-8)
                    nodes_coords[i] += direction * (0.15 - dist) * 0.5
                    nodes_coords[j] -= direction * (0.15 - dist) * 0.5

    G = nx.Graph()

    cs_indices = random.sample(range(N_NODES), N_CS)

    for i, p in enumerate(nodes_coords):
        G.add_node(i, pos=p, is_cs=(i in cs_indices))

    for i in range(N_NODES):
        for j in range(i+1, N_NODES):
            dist = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
            if dist < 0.6:
                G.add_edge(i, j, weight=dist)

    plt.figure(figsize=(16, 12))
    pos = nx.get_node_attributes(G, 'pos')

    node_colors = ['#2ecc71' if G.nodes[n]['is_cs'] else '#3498db' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=650, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=1.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        dist_km = data.get('weight', 0) / 1000.0
        time_m = data.get('time', dist_km / (30.0/60.0)) if 'time' in data else dist_km / (30.0/60.0)
        edge_labels[(u, v)] = f"{dist_km:.2f}km\n{time_m:.1f}m"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

    plt.title(f"SG-GAN Generated Road Network: Dwarka Mor\n({N_NODES} Junctions | {N_CS} CS Nodes | {N_EV} EVs Initialized)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    return G

if __name__ == "__main__":
    generated_graph = generate_dwarka_mor_network()
