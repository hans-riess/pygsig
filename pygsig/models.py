import torch
import torch.nn as nn
from torch_geometric_temporal  import GConvGRU, GConvLSTM,DCRNN
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Temporal GNN models
class GConvGRURegression(nn.Module):
    def __init__(self, num_channels, K):
        super().__init__()
        self.recurrent = GConvGRU(num_channels[0], num_channels[-2], K=K)
        self.lin = nn.Linear(num_channels[-2], num_channels[-1])
                
    def reset_parameters(self):
        for weight in self.recurrent.parameters():
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)
        self.lin.reset_parameters()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:
        x = self.recurrent(x, edge_index, edge_weight)
        x = self.lin(x)
        return torch.sigmoid(x)

class GConvLSTMRegression(nn.Module):
    def __init__(self, num_channels, K):
        super().__init__()
        self.recurrent = GConvLSTM(num_channels[0], num_channels[-2], K=K)
        self.lin = nn.Linear(num_channels[-2], num_channels[-1])
    
    def reset_parameters(self):
        self.recurrent.reset_parameters()
        self.lin.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight=None)->Tensor:
        x, _ = self.recurrent(x, edge_index, edge_weight)
        x = self.lin(x)
        return F.sigmoid(x)


# Static GNN models

class GCNRegression(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.conv = nn.ModuleList()
        self.dropout = nn.Dropout()
        for l in range(self.num_layers):
            self.conv.append(gnn.GCNConv(self.num_channels[l], self.num_channels[l + 1]))
    
    def reset_parameters(self):
        for layer in self.conv:
            layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.sigmoid(x)

class ChebNetRegression(nn.Module):
    def __init__(self, num_channels,K=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.dropout = nn.Dropout()
        self.K = K
        self.conv = nn.ModuleList()
        for l in range(self.num_layers):
            self.conv.append(gnn.ChebConv(self.num_channels[l], self.num_channels[l + 1],self.K,normalization=None))
    
    def reset_parameters(self):
        for layer in self.conv:
            layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.sigmoid(x)

class GATRegression(nn.Module):
    def __init__(self, num_channels,num_heads=1,concat=True):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.dropout = nn.Dropout()
        self.num_heads = num_heads
        self.concat = concat

        self.conv = nn.ModuleList()
        for l in range(self.num_layers):
            if l < self.num_layers-1:
                if self.concat:
                    self.conv.append(gnn.GATConv(in_channels=self.num_channels[l], out_channels=self.num_channels[l + 1],heads=num_heads,concat=True))
                    self.num_channels[l+1]*=num_heads
                else:
                    self.conv.append(gnn.GATConv(in_channels=self.num_channels[l], out_channels=self.num_channels[l + 1],heads=num_heads,concat=False))
            else:
                self.conv.append(gnn.GATConv(in_channels=self.num_channels[l], out_channels=self.num_channels[l + 1],heads=1,concat=False))

    def reset_parameters(self):
        for layer in self.conv:
            layer.reset_parameters()
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return F.sigmoid(x)


# Benchmarks

class MLPRegression(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = len(num_channels) - 1
        self.linear = nn.ModuleList()
        self.dropout = nn.Dropout()
        for l in range(self.num_layers):
            self.linear.append(nn.Linear(self.num_channels[l], self.num_channels[l + 1]))

    def reset_parameters(self):
        for layer in self.linear:
            layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        for i, layer in enumerate(self.linear):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return F.sigmoid(x)

# Our models
class SignatureGAT(nn.Module):
    def __init__(self):
        pass

class SignatureGCN(nn.Module):
    def __init__(self):
        pass