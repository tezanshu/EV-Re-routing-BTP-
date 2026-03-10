import osmnx as ox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


# --- 1. DATA EXTRACTION (Dwarka Mor, New Delhi) ---
def get_map_data():
    lat, lon = 28.6186, 77.0315
    try:
        # Fetching road network for Dwarka Mor area [cite: 3, 65]
        graph = ox.graph_from_point((lat, lon), dist=800, network_type='drive')
        nodes, _ = ox.graph_to_gdfs(graph)
        coords = nodes[['x', 'y']].values
        # Min-max normalization for GAN training (Eq. 3) [cite: 103, 105]
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        norm_coords = -1 + 2 * (coords - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        return torch.FloatTensor(norm_coords), (x_min, x_max, y_min, y_max)
    except:
        return torch.randn(100, 2), (0, 1, 0, 1)


# --- 2. SG-GAN MODELS (Algorithm 1) [cite: 107] ---
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim), nn.Tanh()  # Normalized range [-1, 1]
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


# --- 3. Q-LEARNING AGENT (Algorithm 2) [cite: 185] ---
class EVRoutingAgent:
    def __init__(self, agent_id, actions_map, params):
        self.id = agent_id
        self.q_table = {}  # Initialized to zero (Line 3) [cite: 214]
        self.actions_map = actions_map
        self.params = params
        self.soc = random.uniform(0.3, 1.0)  # Initial SOC (kWh/Total) [cite: 299]
        self.cur = None
        self.dest = None
        self.trip_energy = 0
        self.trip_time = 0

    def get_q(self, s, a):
        return self.q_table.get((s, a), 0.0)

    def select_action(self, state, epsilon):
        # e-greedy policy (Eq. 15, 16) [cite: 276, 278]
        actions = self.actions_map.get(state, [])
        if not actions: return None
        if random.random() < epsilon: return random.choice(actions)
        qs = [self.get_q(state, a) for a in actions]
        return actions[np.argmax(qs)]

    def update_q(self, s, a, r, s_prime, alpha, gamma):
        # Q-value update (Eq. 13) [cite: 263]
        next_actions = self.actions_map.get(s_prime, [])
        max_q_prime = max([self.get_q(s_prime, a_p) for a_p in next_actions]) if next_actions else 0
        self.q_table[(s, a)] = self.get_q(s, a) + alpha * (r + gamma * max_q_prime - self.get_q(s, a))


# --- 4. PHYSICS & REWARD (Eq. 6, 7, 12) [cite: 144, 154, 201] ---
def compute_physics_costs(dist_norm, params):
    # Mechanical to electrical energy conversion and resistance (Eq. 6) [cite: 144]
    v = 10  # m/s
    m = 1500  # kg
    eta = 0.9  # Efficiency
    # Simplified energy calculation based on Eq. 6
    energy_kwh = (1 / eta) * (0.01 * m * 9.81 * v) * (dist_norm * 1000 / v) / 3600000
    time_min = (dist_norm * 1000) / v / 60  # Eq. 7 [cite: 154]
    return energy_kwh, time_min


def calculate_reward(energy, time, reached, params):
    # Reward Function (Eq. 12) [cite: 201]
    congestion = random.uniform(0.1, 0.4)  # delta_ij (Eq. 9) [cite: 163]
    weighted_cost = (params['beta'] * (energy / params['E_norm'])) + \
                    ((1 - params['beta']) * (time / params['T_norm']))
    reward = params['R_base'] - (weighted_cost * params['K_obj']) - \
             (congestion * params['K_cong']) + (reached * params['K_bonus'])
    return reward


# --- 5. MAIN INTEGRATED PIPELINE ---
def main():
    n_nodes, n_agents, noise_dim = 15, 10, 10
    params = {'beta': 0.8, 'E_norm': 10, 'T_norm': 60, 'R_base': 10, 'K_obj': 5,
              'K_cong': 2, 'K_bonus': 100, 'alpha': 0.1, 'gamma': 0.75, 'epsilon': 0.1}

    # STEP 1: SG-GAN Generation (Algorithm 1) [cite: 107]
    real_data, _ = get_map_data()
    netG, netD = Generator(noise_dim, n_nodes * 2), Discriminator(n_nodes * 2)
    optG, optD = optim.Adam(netG.parameters(), lr=0.0002), optim.Adam(netD.parameters(), lr=0.0002)

    for _ in range(201):
        idx = np.random.randint(0, len(real_data), n_nodes)
        real_batch = real_data[idx].view(1, -1)
        optD.zero_grad()
        loss_d = nn.BCELoss()(netD(real_batch), torch.ones(1, 1)) + \
                 nn.BCELoss()(netD(netG(torch.randn(1, noise_dim)).detach()), torch.zeros(1, 1))
        loss_d.backward();
        optD.step()
        optG.zero_grad()
        loss_g = nn.BCELoss()(netD(netG(torch.randn(1, noise_dim))), torch.ones(1, 1))
        loss_g.backward();
        optG.step()

    # STEP 2: Graph Setup with CS Nodes (Section VI)
    nodes_coords = netG(torch.randn(1, noise_dim)).detach().numpy().reshape(-1, 2)
    G = nx.Graph()
    # Designate CS nodes (approximately 25% of nodes as per paper ratio) [cite: 295]
    cs_indices = [3, 7, 12]
    cs_energy_load = {idx: 0.0 for idx in cs_indices}

    for i, p in enumerate(nodes_coords): G.add_node(i, pos=p, is_cs=(i in cs_indices))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
            if d < 0.7: G.add_edge(i, j, weight=d)

    # STEP 3: Multi-Agent Q-Learning (Algorithm 2) [cite: 185]
    actions_map = {n: list(G.neighbors(n)) for n in G.nodes()}
    agents = [EVRoutingAgent(i, actions_map, params) for i in range(n_agents)]
    total_simulation_energy = 0

    for episode in range(500):
        for agent in agents:
            agent.cur, agent.dest = random.choice(list(G.nodes())), random.choice(list(G.nodes()))
            while agent.cur != agent.dest:
                action = agent.select_action(agent.cur, params['epsilon'])
                if action is None: break

                # Check for Charging requirement (SOC < 25%)
                if agent.soc < 0.25 and agent.cur in cs_indices:
                    charge_needed = 1.0 - agent.soc
                    cs_energy_load[agent.cur] += charge_needed
                    agent.soc = 1.0

                dist = G[agent.cur][action]['weight']
                e, t = compute_physics_costs(dist, params)
                reward = calculate_reward(e, t, (action == agent.dest), params)

                agent.update_q(agent.cur, action, reward, action, params['alpha'], params['gamma'])
                agent.soc -= e
                agent.trip_energy += e
                agent.cur = action
            total_simulation_energy += agent.trip_energy

    # --- FINAL OUTPUTS & VISUALIZATION ---
    avg_energy_trip = total_simulation_energy / (n_agents * 500)
    print(f"\n--- Simulation Metrics ---")
    print(f"Average Energy Per Trip: {avg_energy_trip:.6f} kWh")
    for cs_id, load in cs_energy_load.items():
        print(f"Total Energy Distributed at CS Node {cs_id}: {load:.4f} kWh")

    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    colors = ['green' if G.nodes[n]['is_cs'] else 'skyblue' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray', node_size=700)
    plt.title("Weighted Road Network (Green = Charging Stations)")
    plt.show()


if __name__ == "__main__":
    main()