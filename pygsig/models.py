import torch.nn as nn
from torch_geometric_temporal  import GConvGRU, GConvLSTM,DCRNN
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Temporal GNN models
class GConvGRUClassifier(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_targets: int, K: int):
        super().__init__()
        self.recurrent = GConvGRU(in_channels=num_features, out_channels=hidden_channels, K=K)
        self.lin = nn.Linear(hidden_channels, num_targets)
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight=None)->Tensor:
        x = self.recurrent(x, edge_index, edge_weight)
        x = self.lin(x)
        return x

# Static GNN models
class GCNClassifier(nn.Module):
    def __init__(self, num_channels,p_dropout=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.conv = nn.ModuleList()
        self.dropout = nn.Dropout(p=p_dropout)
        for l in range(self.num_layers):
            self.conv.append(gnn.GCNConv(self.num_channels[l], self.num_channels[l + 1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class ChebNetClassifier(nn.Module):
    def __init__(self, num_channels,p_dropout=0.5, K=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.conv = nn.ModuleList()
        self.dropout = nn.Dropout(p=p_dropout)
        self.K = K
        for l in range(self.num_layers):
            self.conv.append(gnn.ChebConv(self.num_channels[l], self.num_channels[l + 1],self.K))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class GCNRegressor(nn.Module):
    def __init__(self, num_channels,p_dropout=0.5):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.conv = nn.ModuleList()
        self.dropout = nn.Dropout(p=p_dropout)
        for l in range(self.num_layers):
            self.conv.append(gnn.GCNConv(self.num_channels[l], self.num_channels[l + 1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.sigmoid(x)


# Baseline models

class MLPClassifier(nn.Module):
    def __init__(self, num_channels,p_dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.linear = nn.ModuleList()
        self.dropout = nn.Dropout(p=p_dropout)
        for l in range(self.num_layers):
            self.linear.append(nn.Linear(self.num_channels[l], self.num_channels[l + 1]))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.linear):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


# Our models
class SignatureGAT(nn.Module):
    def __init__(self):
        pass

class SignatureGCN(nn.Module):
    def __init__(self):
        pass