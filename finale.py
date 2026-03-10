import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy


# --- 1. DATA EXTRACTION (Real World Reference: Dwarka Mor) ---
def get_map_data():
    lat, lon = 28.6186, 77.0315
    print(f"Fetching road network for Dwarka Mor ({lat}, {lon})...")
    try:
        # Fetching driveable road network within 800m
        graph = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values
        # Min-max normalization as per Eq. 3
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        return torch.FloatTensor(norm_coords), (x_min, x_max, y_min, y_max)
    except Exception as e:
        print(f"Error fetching data: {e}. Using synthetic fallback.")
        return torch.randn(100, 2), (0, 1, 0, 1)


# --- 2. SG-GAN MODELS (Algorithm 1)  ---
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Tanh()  # Normalizes to range [-1, 1]
        )

    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()  # Binary classification: Real vs Fake
        )

    def forward(self, x): return self.main(x)


# --- 3. Q-LEARNING AGENT (Algorithm 2) [cite: 3, 4] ---
class EVRoutingAgent:
    def __init__(self, agent_id, actions_map, params):
        self.id = agent_id
        self.q_table = {}  # Qi(s, a) initialized to 0
        self.actions_map = actions_map
        self.params = params
        self.soc = random.uniform(0.3, 1.0)  # Initial SOC [cite: 5]
        self.cur = None
        self.dest = None

    def get_q(self, s, a):
        return self.q_table.get((s, a), 0.0)

    def select_action(self, state, epsilon):
        # e-greedy policy (Eq 15, 16) [cite: 5]
        actions = self.actions_map.get(state, [])
        if not actions: return None
        if random.random() < epsilon: return random.choice(actions)
        qs = [self.get_q(state, a) for a in actions]
        return actions[np.argmax(qs)]

    def update_q(self, s, a, r, s_prime, alpha, gamma):
        # Q-Value Update Rule (Eq 13) [cite: 5]
        next_actions = self.actions_map.get(s_prime, [])
        max_q_prime = max([self.get_q(s_prime, a_p) for a_p in next_actions]) if next_actions else 0
        self.q_table[(s, a)] = self.get_q(s, a) + alpha * (r + gamma * max_q_prime - self.get_q(s, a))


# --- 4. REWARD & PHYSICS CALCULATION (Eq. 6, 7, 12) [cite: 3, 4] ---
def compute_costs_and_reward(dist, reached, params):
    # Constants from Table I [cite: 5]
    v = 10  # Velocity in m/s
    eta = 0.9  # Efficiency
    rho = 1.225  # Air density
    Cd = 0.3  # Drag coeff
    A = 2.5  # Frontal area
    Cr = 0.01  # Rolling resistance
    m = 1500  # Mass
    g = 9.81

    # Energy Consumption Eci (Eq 6)
    energy = (1 / eta) * (0.5 * rho * Cd * A * v ** 2 + Cr * m * g * v) * (dist * 1000 / v)
    # Travel Time Tci (Eq 7)
    time = (dist * 1000) / v

    # Congestion Factor (Eq 9)
    congestion = random.uniform(0.1, 0.4)

    # Reward Function (Eq 12)
    weighted_cost = (params['beta'] * (energy / params['E_norm'])) + \
                    ((1 - params['beta']) * (time / params['T_norm']))

    reward = params['R_base'] - (weighted_cost * params['K_obj']) - \
             (congestion * params['K_cong']) + (reached * params['K_bonus'])
    return reward


# --- 5. MAIN EXECUTION PIPELINE ---
def main():
    # Parameters from Tables I & II [cite: 5]
    n_nodes, n_agents, noise_dim = 15, 5, 10
    params = {'beta': 0.8, 'E_norm': 50000, 'T_norm': 3600, 'R_base': 10, 'K_obj': 5,
              'K_cong': 2, 'K_bonus': 100, 'alpha': 0.1, 'gamma': 0.75, 'epsilon': 0.1}

    # STEP A: Train SG-GAN (Algorithm 1)
    real_data, bounds = get_map_data()
    netG, netD = Generator(noise_dim, n_nodes * 2), Discriminator(n_nodes * 2)
    optG, optD = optim.Adam(netG.parameters(), lr=0.0002), optim.Adam(netD.parameters(), lr=0.0002)

    print("Executing Algorithm 1 (SG-GAN Road Generation)...")
    for epoch in range(201):
        idx = np.random.randint(0, len(real_data), n_nodes)
        real_batch = real_data[idx].view(1, -1)
        # Train Discriminator (D-step)
        optD.zero_grad()
        loss_d = nn.BCELoss()(netD(real_batch), torch.ones(1, 1)) + \
                 nn.BCELoss()(netD(netG(torch.randn(1, noise_dim)).detach()), torch.zeros(1, 1))
        loss_d.backward();
        optD.step()
        # Train Generator (G-step)
        optG.zero_grad()
        loss_g = nn.BCELoss()(netD(netG(torch.randn(1, noise_dim))), torch.ones(1, 1))
        loss_g.backward();
        optG.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | D-Loss: {loss_d.item():.4f} | G-Loss: {loss_g.item():.4f}")

    # STEP B: Construct Weighted Graph (Eq 11)
    nodes = netG(torch.randn(1, 10)).detach().numpy().reshape(-1, 2)
    G = nx.Graph()
    for i, pos in enumerate(nodes): G.add_node(i, pos=pos)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.linalg.norm(nodes[i] - nodes[j])
            if d < 0.6: G.add_edge(i, j, weight=d)

    # STEP C: Train EV Agents (Algorithm 2)
    print("Executing Algorithm 2 (Multi-Agent Q-Learning)...")
    actions_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    agents = [EVRoutingAgent(i, actions_map, params) for i in range(n_agents)]
    for a in agents:
        a.cur, a.dest = random.choice(list(G.nodes())), random.choice(list(G.nodes()))

    for episode in range(1000):  # Training episodes [cite: 5]
        for agent in agents:
            if agent.cur == agent.dest: continue
            action = agent.select_action(agent.cur, params['epsilon'])
            if action is None: continue

            reward = compute_costs_and_reward(G[agent.cur][action]['weight'],
                                              (action == agent.dest), params)
            agent.update_q(agent.cur, action, reward, action, params['alpha'], params['gamma'])
            agent.cur = action

    # STEP D: FINAL RESULTS & VISUALIZATION
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=600)
    plt.title("Paper Replication: SG-GAN Road Network & Optimized EV Routing (Dwarka Mor)")
    plt.show()
    print("Simulation Complete.")


if __name__ == "__main__":
    main()