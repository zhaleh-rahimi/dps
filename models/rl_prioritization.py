import numpy as np
import random


class InventoryRLAgent:
    def __init__(self, state_size=5, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def train(self, clusters, inventory, demand):
        for episode in range(100):
            state = random.choice(range(self.state_size))
            action = random.choice(range(self.action_size))
            reward = self._compute_reward(action, inventory, demand)
            self.q_table[state, action] += 0.1 * (reward - self.q_table[state, action])

    def _compute_reward(self, action, inventory, demand):
        return np.random.rand()
