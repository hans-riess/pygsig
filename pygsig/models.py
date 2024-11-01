import torch
import torch.nn as nn
from torch_geometric_temporal  import GConvGRU, GConvLSTM,DCRNN
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# RNN models
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
        _, c = self.recurrent(x, edge_index, edge_weight)
        c = self.lin(c)
        return F.sigmoid(c)

class DCRNNRegression(nn.Module):
    def __init__(self,num_channels,K):
        super().__init__()
        self.recurrent = DCRNN(num_channels[0], num_channels[-2], K=K)
        self.lin = nn.Linear(num_channels[-2], num_channels[-1])
    
    def reset_parameters(self):
        self.recurrent.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight=None)->Tensor:
        x = self.recurrent(x, edge_index, edge_weight)
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
        return F.sigmoid(x)

class GCNClassification(nn.Module):
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
        return x

class SigGCNClassification(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = gnn.GCNConv(num_channels[0], num_channels[1])
        self.lin = nn.Linear(num_channels[-2], num_channels[-1])
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        x = self.conv(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.lin(x)
        return x

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
        return F.sigmoid(x)
    
class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MLPClassification(nn.Module):
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
        return x