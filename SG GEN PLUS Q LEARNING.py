import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.stats import entropy


# --- 1. DATA EXTRACTION (Dwarka Mor) ---
def get_dwarka_mor_data():
    lat, lon = 28.6186, 77.0315
    try:
        graph = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        # Normalization as per Eq 3
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        return torch.FloatTensor(norm_coords), (x_min, x_max, y_min, y_max)
    except:
        return torch.randn(100, 2), (0, 1, 0, 1)


# --- 2. SG-GAN MODELS (Algorithm 1) ---
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(nn.Linear(noise_dim, 128), nn.ReLU(), nn.Linear(128, output_dim), nn.Tanh())

    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x): return self.main(x)


# --- 3. Q-LEARNING AGENT (Algorithm 2) ---
class EVRoutingAgent:
    def __init__(self, agent_id, actions_map, params):
        self.id = agent_id
        self.q_table = {}  # Qi(s, a) initialized to 0
        self.actions_map = actions_map
        self.params = params
        self.soc = random.uniform(0.4, 0.9)  # Initial SOC
        self.cur = None
        self.dest = None

    def get_q(self, s, a):
        return self.q_table.get((s, a), 0.0)

    def select_action(self, state, epsilon):
        # Epsilon-greedy policy (Eq 15, 16)
        actions = self.actions_map.get(state, [])
        if not actions: return None
        if random.random() < epsilon: return random.choice(actions)
        qs = [self.get_q(state, a) for a in actions]
        return actions[np.argmax(qs)]

    def update_q(self, s, a, r, s_prime, alpha, gamma):
        # Q-Value Update (Eq 13)
        next_actions = self.actions_map.get(s_prime, [])
        max_q_prime = max([self.get_q(s_prime, a_p) for a_p in next_actions]) if next_actions else 0
        self.q_table[(s, a)] = self.get_q(s, a) + alpha * (r + gamma * max_q_prime - self.get_q(s, a))


# --- 4. PHYSICS & REWARD (Eq. 6, 7, 12) ---
def calculate_physics_costs(dist_norm, params):
    # Energy Consumption Eci (Eq 6)
    # Travel Time Tci (Eq 7)
    energy = dist_norm * 50
    time = dist_norm * 100
    return energy, time


def compute_reward(energy, time, reached, local_congestion, p):
    # Reward Function (Eq 12)
    weighted_cost = (p['beta'] * (energy / p['E_norm'])) + ((1 - p['beta']) * (time / p['T_norm']))
    # Penalty for congestion
    reward = p['R_base'] - (weighted_cost * p['K_obj']) - (local_congestion * p['K_cong']) + (reached * p['K_bonus'])
    return reward


# --- 5. MAIN EXECUTION PIPELINE ---
def main():
    # Parameters from papers
    noise_dim, n_nodes, n_agents = 10, 15, 5
    params = {'beta': 0.5, 'E_norm': 1000, 'T_norm': 3600, 'R_base': 10, 'K_obj': 5,
              'K_cong': 2, 'K_bonus': 100, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.2}

    # STEP A: SG-GAN Generation (Algorithm 1)
    real_data, _ = get_dwarka_mor_data()
    netG, netD = Generator(noise_dim, n_nodes * 2), Discriminator(n_nodes * 2)
    optG, optD = optim.Adam(netG.parameters(), lr=0.001), optim.Adam(netD.parameters(), lr=0.001)

    print("Training SG-GAN (Algorithm 1)...")
    for epoch in range(201):
        idx = np.random.randint(0, len(real_data), n_nodes)
        real_b = real_data[idx].view(1, -1)
        # Train Discriminator
        optD.zero_grad()
        loss_d = nn.BCELoss()(netD(real_b), torch.ones(1, 1)) + \
                 nn.BCELoss()(netD(netG(torch.randn(1, 10)).detach()), torch.zeros(1, 1))
        loss_d.backward();
        optD.step()
        # Train Generator
        optG.zero_grad()
        loss_g = nn.BCELoss()(netD(netG(torch.randn(1, 10))), torch.ones(1, 1))
        loss_g.backward();
        optG.step()

    # STEP B: Construct Weighted Graph (Eq 11)
    nodes_final = netG(torch.randn(1, noise_dim)).detach().numpy().reshape(-1, 2)
    G = nx.Graph()
    for i, p in enumerate(nodes_final): G.add_node(i, pos=p)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.linalg.norm(nodes_final[i] - nodes_final[j])
            if d < 0.5:  # Spatial proximity threshold
                G.add_edge(i, j, weight=d)

    # STEP C: Multi-Agent Q-Learning (Algorithm 2)
    print("Training EV Agents (Algorithm 2)...")
    actions_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    agents = [EVRoutingAgent(i, actions_map, params) for i in range(n_agents)]
    for a in agents:
        a.cur, a.dest = random.choice(list(G.nodes())), random.choice(list(G.nodes()))

    for episode in range(200):
        for agent in agents:
            if agent.cur == agent.dest: continue
            action = agent.select_action(agent.cur, params['epsilon'])
            if action is None: continue

            # Observe environment
            dist = G[agent.cur][action]['weight']
            e, t = calculate_physics_costs(dist, params)
            cong = random.uniform(0.1, 0.5)  # Simulated delta_ij (Eq 9)

            reached = (action == agent.dest)
            reward = compute_reward(e, t, reached, cong, params)

            # Update and move
            agent.update_q(agent.cur, action, reward, action, params['alpha'], params['gamma'])
            agent.cur = action

    # STEP D: Final Results Visualization
    plt.figure(figsize=(10, 6))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='blue', width=2)
    plt.title("Dwarka Mor: SG-GAN Generated Road Network & Trained EV Agents")
    plt.show()


if __name__ == "__main__":
    main()