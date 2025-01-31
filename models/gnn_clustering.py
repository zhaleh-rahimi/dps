import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class CustomerClustering:
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=8):
        self.model = GNNModel(input_dim, hidden_dim, output_dim)

    def cluster_customers(self, customer_data):
        edge_index = customer_data['edges']
        features = customer_data['features']
        embeddings = self.model(features, edge_index)
        clusters = KMeans(n_clusters=4).fit_predict(embeddings.detach().numpy())
        return clusters
