import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# --- 1. PARAMETERS (From Table I & II) ---
params = {
    'rho': 1.225, 'Cd': 0.3, 'A': 2.5, 'Cr': 0.01, 'm': 1500,
    'g': 9.81, 'eta': 0.9, 'theta': 5, 'Bmax': 10.0, 'Th': 0.25,
    'alpha': 0.1, 'gamma': 0.75, 'beta': 0.8, 'epsilon': 0.1,
    'K_obj': 5, 'K_cong': 2, 'K_bonus': 100, 'R_base': 10,
    'E_norm': 10.0, 'T_norm': 120.0  # Normalization factors [cite: 13]
}


# --- 2. SG-GAN MODELS (Fig 1 Architecture) ---
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
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


# --- 3. CORE LOGIC FUNCTIONS ---
def get_map_data():
    """Extract real data for training reference (Algorithm 1)"""
    lat, lon = 28.6186, 77.0315  # Dwarka Mor Coordinates [cite: 1]
    graph = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
    nodes, _ = ox.graph_to_gdfs(graph)
    coords = nodes[['x', 'y']].values
    # Normalization to [-1, 1] [cite: 8]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    return -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]


def compute_physics(dist_norm, p):
    """Energy (Eq 6) and Time (Eq 7) Calculation [cite: 8, 10]"""
    v = 10  # Velocity m/s (approx 36 km/h) [cite: 14]
    # Energy Eci in kWh
    energy = (1 / p['eta']) * (0.5 * p['rho'] * p['Cd'] * p['A'] * v ** 2 +
                               p['Cr'] * p['m'] * p['g'] * v +
                               p['m'] * p['g'] * dist_norm * np.sin(np.radians(p['theta']))) * \
             (dist_norm * 1000 / v) / 3600000
    time = (dist_norm * 1000) / v / 60  # Minutes
    return energy, time


# --- 4. MAIN SIMULATION ---
def main():
    # Algorithm 1: SG-GAN Road Network Generation
    real_coords = torch.FloatTensor(get_map_data())
    n_nodes = 29  # As per Table II [cite: 13]
    gen = Generator(10, n_nodes * 2)
    disc = Discriminator(n_nodes * 2)
    optimizer_g = optim.Adam(gen.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(disc.parameters(), lr=0.0002)

    print("Training SG-GAN (Algorithm 1)...")
    for epoch in range(201):  # Replicating Iteration 200 convergence [cite: 1, 11]
        idx = np.random.randint(0, len(real_coords), n_nodes)
        real_batch = real_coords[idx].view(1, -1)
        noise = torch.randn(1, 10)
        fake_batch = gen(noise)

        # Update Discriminator
        optimizer_d.zero_grad()
        d_loss = nn.BCELoss()(disc(real_batch), torch.ones(1, 1)) + \
                 nn.BCELoss()(disc(fake_batch.detach()), torch.zeros(1, 1))
        d_loss.backward();
        optimizer_d.step()

        # Update Generator
        optimizer_g.zero_grad()
        g_loss = nn.BCELoss()(disc(fake_batch), torch.ones(1, 1))
        g_loss.backward();
        optimizer_g.step()

    # Build the Weighted Graph (G)
    nodes = gen(torch.randn(1, 10)).detach().numpy().reshape(-1, 2)
    G = nx.Graph()
    cs_nodes = random.sample(range(n_nodes), 7)  # 7 CS nodes as per Fig 2

    for i, p in enumerate(nodes):
        G.add_node(i, pos=p, is_cs=(i in cs_nodes), soc_load=0)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.linalg.norm(nodes[i] - nodes[j])
            if dist < 0.6:  # Spatial threshold for connectivity
                e, t = compute_physics(dist, params)
                G.add_edge(i, j, weight=dist, energy=e, time=t)

    # Algorithm 2: Q-Learning for EV Routing
    print("Executing Multi-Agent Q-Learning (Algorithm 2)...")
    q_table = {}
    n_agents = 5  # 5 EVs for testing [cite: 12]

    for episode in range(1001):
        for agent in range(n_agents):
            curr = random.choice(list(G.nodes()))
            dest = random.choice(list(G.nodes()))
            soc = random.uniform(0.3, 1.0)  # Initial SOC [cite: 12]

            while curr != dest:
                neighbors = list(G.neighbors(curr))
                # Epsilon-greedy selection (Eq 15, 16) [cite: 9]
                if random.random() < params['epsilon']:
                    action = random.choice(neighbors)
                else:
                    action = neighbors[np.argmax([q_table.get((curr, a), 0) for a in neighbors])]

                # Observe transitions
                edge = G[curr][action]
                reached = (action == dest)
                cong = random.uniform(0.1, 0.4)  # Simulated congestion delta [cite: 12]

                # Reward Function (Eq 12) [cite: 7, 9]
                w_cost = (params['beta'] * (edge['energy'] / params['E_norm'])) + \
                         ((1 - params['beta']) * (edge['time'] / params['T_norm']))
                reward = params['R_base'] - (w_cost * params['K_obj']) - \
                         (cong * params['K_cong']) + (reached * params['K_bonus'])

                # Q-Update (Eq 13) [cite: 9]
                next_q = max([q_table.get((action, a), 0) for a in G.neighbors(action)], default=0)
                q_table[(curr, action)] = q_table.get((curr, action), 0) + \
                                          params['alpha'] * (
                                                      reward + params['gamma'] * next_q - q_table.get((curr, action),
                                                                                                      0))

                # SOC Management [cite: 12]
                soc -= edge['energy']
                if soc < params['Th'] and G.nodes[action]['is_cs']:
                    soc = 1.0  # Recharged

                curr = action

    # Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    colors = ['green' if G.nodes[n]['is_cs'] else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=600, edge_color='blue')
    plt.title("SG-GAN Generated Road Network (Dwarka Mor)\nGreen: Charging Stations | Blue: Junctions")
    plt.show()


if __name__ == "__main__":
    main()