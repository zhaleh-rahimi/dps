import gym
import numpy as np
from gym import spaces


class InventoryEnv(gym.Env):
    def __init__(self):
        super(InventoryEnv, self).__init__()
        self.inventory_level = 100
        self.action_space = spaces.Discrete(3)  # Actions: [Reduce, Maintain, Increase Stock]
        self.observation_space = spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.inventory_level -= 10  # Reduce stock
        elif action == 2:
            self.inventory_level += 10  # Increase stock

        reward = -abs(self.inventory_level - 100)  # Reward for maintaining optimal inventory
        done = self.inventory_level <= 0 or self.inventory_level >= 200
        return np.array([self.inventory_level]), reward, done, {}

    def reset(self):
        self.inventory_level = 100
        return np.array([self.inventory_level])
