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
from pygsig.graph import GeometricGraph, CustomStaticGraphTemporalSignal

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

class SignatureFeatures(T.BaseTransform):
    def __init__(self, num_node_features,sig_depth=3,normalize=True,log_signature=False):
        super().__init__()
        self.sig_depth = sig_depth
        self.normalize = normalize
        if log_signature:
            self.signature = LogSignature(depth=sig_depth)
        else:
            self.signature = Signature(depth=sig_depth)

    def forward(self, dataset: CustomStaticGraphTemporalSignal) -> Data:
        y = dataset[-1].y
        x_seq = torch.zeros([dataset.num_nodes,dataset.snapshot_count,dataset.num_node_features])
        pos = dataset[-1].pos
        for time,feature in enumerate(dataset.features):
            x_seq[:,time,:] = torch.tensor(feature)
        x = self.signature(x_seq)       
        if self.normalize:
            std_x = torch.std(x,dim=0)
            mean_x = torch.mean(x,dim=0)
            x = (x - mean_x)/std_x
        dataset_static = GeometricGraph(x=x,y=y,edge_index=dataset.edge_index,edge_weight=dataset.edge_weight,pos=pos)
        return dataset_static