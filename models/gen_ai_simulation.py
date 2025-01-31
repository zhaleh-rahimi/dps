import torch
import torch.nn as nn


class DemandSimulator:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def generate_scenarios(self, demand_data):
        return self.model(torch.tensor(demand_data, dtype=torch.float32)).detach().numpy()
