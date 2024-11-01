from typing import Optional
import math
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
import torch_geometric.transforms as T
from torch_geometric.data import Data
from signatory import multi_signature_combine
from signatory import extract_signature_term, signature_channels
from signatory import Signature, LogSignature
from pygsig.graph import StaticGraphTemporalSignal, GeometricGraph
import numpy as np

class SignatureFeatures(T.BaseTransform):
    def __init__(self, sig_depth=3, normalize=True, log_signature=False, time_augment=False, lead_lag=False):
        super().__init__()
        self.sig_depth = sig_depth
        self.normalize = normalize
        self.time_augment = time_augment
        self.lead_lag = lead_lag
        if log_signature:
            self.signature = LogSignature(depth=sig_depth)
        else:
            self.signature = Signature(depth=sig_depth)

    def lead_lag_transform(self, x_seq):
        batch_size, seq_length, feature_dim = x_seq.size()

        # Initialize an empty tensor for the lead-lag path
        lead_lag_seq = torch.zeros(batch_size, 2 * seq_length - 1, feature_dim * 2, device=x_seq.device, dtype=x_seq.dtype)

        # Even indices (lag equals lead)
        lead_lag_seq[:, 0::2, :] = torch.cat((x_seq, x_seq), dim=-1)

        # Odd indices (lead is one step ahead)
        lead_lag_seq[:, 1::2, :] = torch.cat((x_seq[:, 1:, :], x_seq[:, :-1, :]), dim=-1)

        return lead_lag_seq
    
    def time_augment_transform(self, x_seq):
        batch_size, seq_length, feature_dim = x_seq.size()
        # Create a time vector normalized between 0 and 1
        time_vector = torch.linspace(0, 1, steps=seq_length, device=x_seq.device, dtype=x_seq.dtype)
        # Repeat and reshape to match x_seq dimensions
        time_feature = time_vector.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        # Concatenate along the feature dimension
        x_aug = torch.cat([x_seq, time_feature], dim=-1)
        return x_aug

    def forward(self, dataset: StaticGraphTemporalSignal) -> Data:
        y = dataset[-1].y
        pos = dataset[-1].pos

        # Initialize the feature sequence
        x_seq = torch.zeros([dataset.num_nodes, dataset.snapshot_count, dataset.num_node_features])
        for time, feature in enumerate(dataset.features):
            # Check if `feature` is a numpy array, and if so, convert it
            if isinstance(feature, np.ndarray):
                feature = torch.from_numpy(feature).float()  # Convert to float tensor if needed
            x_seq[:, time, :] = feature
                
        # Apply lead-lag if enabled
        if self.lead_lag:
            x_seq = self.lead_lag_transform(x_seq)
        
        # Apply time augmentation if enabled
        if self.time_augment:
            x_seq = self.time_augment_transform(x_seq)

        # Apply signature transformation
        x = self.signature(x_seq)

        # Normalize if required
        if self.normalize:
            std_x = torch.std(x, dim=0)
            mean_x = torch.mean(x, dim=0)
            x = (x - mean_x) / std_x

        # Create the static graph dataset with transformed features
        dataset_static = GeometricGraph(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static

#----------------------------------------------------------------------------------------------------------------
# STILL A WORK IN PROGRESS
#----------------------------------------------------------------------------------------------------------------

class StatFeatures(T.BaseTransform):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, dataset: StaticGraphTemporalSignal) -> Data:
        y = dataset[-1].y
        pos = dataset[-1].pos
        data_matrix = torch.jstack([ dataset[t].x for t in range(dataset.snapshot_count)])
        x = torch.stack([torch.quantile(data_matrix,dim=0,q=0.00),
                            torch.quantile(data_matrix,dim=0,q=0.25),
                            torch.quantile(data_matrix,dim=0,q=0.50),
                            torch.quantile(data_matrix,dim=0,q=0.75),
                            torch.quantile(data_matrix,dim=0,q=1.00)],dim=1).reshape(-1,5*dataset.num_node_features)  
        if self.normalize:
            std_x = torch.std(x,dim=0)
            mean_x = torch.mean(x,dim=0)
            x = (x - mean_x)/std_x
        dataset_static = GeometricGraph(x=x,y=y,edge_index=dataset.edge_index,edge_weight=dataset.edge_weight,pos=pos)
        return dataset_static

class Linear(nn.Module):
    def __init__(self,in_channels,out_channels,depth,bias=True):
        super(Linear, self).__init__()
        self.depth = depth
        self.in_features = in_channels
        self.out_features = out_channels
        self.weight = nn.parameter.Parameter(torch.empty((out_channels, in_channels)))
        if bias == True:
            self.bias = nn.parameter.Parameter(torch.empty([out_channels for _ in range(depth)]))
        else:
            self.bias = None
        self.einsum_exp = self.einsum_expression()
        self.reset_parameters()
    
    def einsum_expression(self):
        from string import ascii_lowercase
        if 2 * self.depth > 26:
            raise ValueError("Depth must be 13 or less due to alphabet limit.")
        main_tensor_indices = ascii_lowercase[self.depth:2*self.depth]  # Continue where matrices indices end
        result_indices = ascii_lowercase[:self.depth]
        matrices_indices = []
        for i in range(self.depth):
            matrices_indices.append(ascii_lowercase[i] + main_tensor_indices[i] )
        einsum_expression = f"{main_tensor_indices}," + ",".join(matrices_indices) + f"->{result_indices}"
        return einsum_expression

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self,x):
        if self.bias == True:
            return torch.einsum(self.einsum_exp, x, *[self.weight for _ in range(self.depth)]) + self.bias
        else: 
            return torch.einsum(self.einsum_exp, x, *[self.weight for _ in range(self.depth)])


class SignatureLinear(nn.Module):
    def __init__(self,in_channels,out_channels,depth,bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.ModuleList([Linear(in_channels,out_channels,d,bias) for d in range(1,depth+1)])
        self.depth = depth
    def forward(self,x):
        outputs = []
        for linear_map in self.linear:
            z = extract_signature_term(x,self.in_channels,linear_map.depth)
            z = z.reshape(*linear_map.depth*[self.in_channels])
            z = linear_map(z)
            outputs.append(torch.flatten(z))
        return torch.concat(outputs,dim=0)

class SignatureAggregation(gnn.aggr.Aggregation):
    def __init__(self, in_channels: int, depth: int, inverse: bool = False, scalar_term: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.inverse = inverse
        self.scalar_term = scalar_term
    
    def signature_combine(self,x: Tensor)->Tensor:
        tensor_list = [x[i].unsqueeze(dim=0) for i in range(x.shape[0])]
        return multi_signature_combine(tensor_list, input_channels=self.in_channels, depth=self.depth)

    def forward(self,x: Tensor,index: Optional[Tensor] = None, 
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2,
                max_num_elements: Optional[int] = None
    ) -> Tensor:
        
        expected_channels = signature_channels(self.in_channels, self.depth, self.scalar_term)
        if x.shape[-1] != expected_channels:
            raise ValueError(f"Input tensor channels {x.shape[-1]} do not match the expected signature channels {expected_channels}")
        
        unique_indices = index.unique()
        aggregated = []
        for idx in unique_indices:
            group = x[index == idx]
            aggregated.append(self.signature_combine(group))
        return torch.stack(aggregated, dim=dim).squeeze()

    def reset_parameters(self):
        pass