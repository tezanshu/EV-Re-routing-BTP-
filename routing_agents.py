import numpy as np
import random
from collections import defaultdict
import networkx as nx


class QLearningAgent:
    def __init__(self, agent_id, G, start_node, dest_node, initial_soc, memory=None):
        self.agent_id = agent_id
        self.G = G
        self.s = start_node
        self.dest = dest_node
        self.soc = initial_soc
        self.max_soc = 10.0
        self.critical_soc = 2.5

        self.Q = memory if memory is not None else defaultdict(lambda: defaultdict(float))

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.total_energy = 0
        self.total_time = 0
        self.path = [self.s]

    def get_actions(self, state):
        return list(self.G.neighbors(state))

    def choose_action(self, state):
        actions = self.get_actions(state)
        if not actions:
            return state

        if random.random() < self.epsilon:
            return random.choice(actions)

        best_a = actions[0]
        max_q = self.Q[state][best_a]
        for a in actions[1:]:
            q = self.Q[state][a]
            if q > max_q:
                max_q = q
                best_a = a
        return best_a

    def get_reward(self, s, a, route_energy, route_time, local_congestion, global_congestion):
        r_base = 10.0

        e_norm = 0.5
        t_norm = 120.0

        beta = 0.8
        obj_penalty = beta * (route_energy / e_norm) + (1 - beta) * (route_time / t_norm)

        cong_penalty = (local_congestion / 100.0) * 5.0

        reward = r_base - obj_penalty - cong_penalty

        if a == self.dest:
            reward += 1000.0

        if self.soc <= 0:
            reward -= 500.0

        return reward

    def step(self, global_congestion_map, cs_status):
        if self.s == self.dest or self.soc <= 0:
            return True, 0, 0

        if self.G.nodes[self.s].get('is_cs', False) and self.soc < (self.max_soc * 0.8):
            charge_amount = self.max_soc - self.soc
            charge_time = (charge_amount / 50.0) * 3600
            self.soc = self.max_soc
            self.total_time += charge_time
            self.Q[self.s]['CHARGE'] += self.alpha * (50 + self.gamma * max(self.Q[self.s].values()) - self.Q[self.s]['CHARGE'])
            return False, 0, charge_time

        a = self.choose_action(self.s)

        edge_data = self.G.get_edge_data(self.s, a)
        distance = edge_data['weight']
        base_time = edge_data['time']
        capacity = edge_data.get('capacity', 20)

        local_cong = global_congestion_map.get((self.s, a), 0)

        delay = (local_cong / capacity) * base_time if local_cong > capacity * 0.5 else 0
        actual_time = base_time + delay

        energy_kwh = (distance / 1000.0) * 0.15 * (1 + (local_cong / capacity) * 0.2)

        self.soc -= energy_kwh
        self.total_energy += energy_kwh
        self.total_time += actual_time

        reward = self.get_reward(self.s, a, energy_kwh, actual_time, local_cong, 0)

        best_next_q = 0
        if a != self.dest:
            next_actions = self.get_actions(a)
            if next_actions:
                best_next_q = max(self.Q[a][na] for na in next_actions)

        self.Q[self.s][a] += self.alpha * (reward + self.gamma * best_next_q - self.Q[self.s][a])

        self.s = a
        self.path.append(a)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.s == self.dest, energy_kwh, actual_time


class MultiAgentRouter:
    def __init__(self, G, n_evs=50, beta=0.8):
        self.G = G
        self.n_evs = n_evs
        self.beta = beta
        self.agents = []
        self.global_q_table = defaultdict(lambda: defaultdict(float))

        nodes = list(G.nodes())
        for i in range(n_evs):
            s, d = random.sample(nodes, 2)
            soc = random.uniform(3.0, 10.0)
            agent = QLearningAgent(i, G, s, d, soc, memory=self.global_q_table)
            self.agents.append(agent)

    def map_bipartite_cs(self, agent_pos, agent_soc):
        cs_nodes = [n for n, d in self.G.nodes(data=True) if d.get('is_cs', False)]
        best_cs = None
        min_cost = float('inf')

        for cs in cs_nodes:
            try:
                path = nx.shortest_path(self.G, agent_pos, cs, weight='weight')
                dist = nx.path_weight(self.G, path, weight='weight')

                energy_req = (dist / 1000.0) * 0.15
                wait_time = 0

                cost = self.beta * (energy_req / 0.5) + (1 - self.beta) * (wait_time / 10)

                if cost < min_cost and agent_soc > energy_req:
                    min_cost = cost
                    best_cs = cs
            except nx.NetworkXNoPath:
                continue

        return best_cs

    def train(self, epochs=2000):
        print(f"Training {self.n_evs} Multi-Agents over {epochs} epochs...")
        history = []

        for epoch in range(epochs):
            nodes = list(self.G.nodes())
            for agent in self.agents:
                agent.s, agent.dest = random.sample(nodes, 2)
                agent.soc = random.uniform(3.0, 10.0)
                agent.path = [agent.s]
                agent.total_energy = 0
                agent.total_time = 0

            active_agents = set(self.agents)
            global_congestion = defaultdict(int)
            epoch_energy = 0

            steps = 0
            while active_agents and steps < 100:
                steps += 1
                completed = []

                for agent in active_agents:
                    if agent.soc <= agent.critical_soc * 1.5 and not self.G.nodes[agent.s].get('is_cs', False):
                        target_cs = self.map_bipartite_cs(agent.s, agent.soc)
                        if target_cs:
                            agent.dest = target_cs

                    prev_s = agent.s
                    done, e_cost, t_cost = agent.step(global_congestion, None)
                    epoch_energy += e_cost

                    if not done and prev_s != agent.s:
                        global_congestion[(prev_s, agent.s)] += 1

                    if done or agent.soc <= 0:
                        completed.append(agent)

                for c in completed:
                    active_agents.remove(c)

                for k in global_congestion:
                    global_congestion[k] = max(0, global_congestion[k] - 0.5)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Avg Energy: {epoch_energy/self.n_evs:.2f} kWh | Q-Table Size: {len(self.global_q_table)}")
            history.append(epoch_energy)

        print("Training Completed.")
        return history
