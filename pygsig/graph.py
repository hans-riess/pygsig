import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch_geometric_temporal as tgnn
from typing import Sequence,Union
from sklearn.utils.class_weight import compute_class_weight


Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]
Positions = Union[np.ndarray, None]

class GraphKernel(T.BaseTransform):
    def __init__(self,bandwith=None):
        self.distances = T.Distance(norm=False)
        self.bandwidth = bandwith
    
    def forward(self, graph: Data)->Data:
        graph = self.distances(graph)
        if self.bandwidth == None:
            self.bandwidth = torch.std(graph.edge_attr)
        graph.edge_attr = torch.exp(-torch.sum(graph.edge_attr**2,dim=-1)/self.bandwidth**2).float()
        return graph

class GeometricGraph(Data):
    def __init__(self, edge_index, x, y, edge_weight,pos):
        super().__init__(edge_index=edge_index, edge_attr=edge_weight,x=x,y=y,pos=pos)
        

class StaticGraphTemporalSignal(tgnn.signal.StaticGraphTemporalSignal):
    def __init__(self,
                edge_index: Edge_Index,
                edge_weight: Edge_Weight,
                features: Node_Features,
                targets: Targets,
                positions: Positions = None,
                **kwargs: Additional_Features
    ):
        super().__init__(edge_index, edge_weight, features, targets)
        self.positions = positions
        self.num_nodes = self.features[0].shape[0]
        self.num_node_features = self.features[0].shape[-1]
        self.y = self._get_target(-1)
        self.graph = GeometricGraph(edge_index=edge_index,edge_weight=edge_weight,pos=positions,x=None,y=None)
    
    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            return torch.FloatTensor(self.targets[time_index])
    
    def _get_positions(self,time_index: int):
        if self.positions[time_index] is None:
            return self.positions[time_index]
        else:
            return torch.DoubleTensor(self.positions[time_index])
    
    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                self.positions[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            pos = self._get_positions(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, pos=pos, **additional_features)
        return snapshot

    def _get_feature_matrix(self,numpy=False):
        X = torch.zeros([self.num_nodes,self.snapshot_count,self.num_node_features])
        y = torch.zeros([self.snapshot_count])
        for time in range(self.snapshot_count):
            X[:,time,:] = torch.tensor(self.features[time])
            y[time] = self.targets[time]
        if numpy:
            return X.numpy(),y.numpy()
        else:
            return X,y

def split_nodes(num_nodes, num_splits,test_ratio = 0.8, seed=29):
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    splits = []
    for i in range(num_splits):
        nontrain_indices = indices[i::num_splits]  # disjoint test set for each split
        train_indices = np.setdiff1d(indices, nontrain_indices)
        # nontrain indicies split into testing and validation
        test_indices = nontrain_indices[:int(test_ratio *len(nontrain_indices))] 
        eval_indices = nontrain_indices[int(test_ratio *len(nontrain_indices)):]
        splits.append((train_indices, eval_indices, test_indices))
    return splits