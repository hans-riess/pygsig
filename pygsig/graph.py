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

class KernelKNNGraph(T.BaseTransform):
    def __init__(self,k: int, bandwidth: float):
        super().__init__()
        self.k = k
        self.bandwidth = bandwidth
        self.knn_graph = T.KNNGraph(self.k,force_undirected=True,loop=False,num_workers=4)
        self.transform = T.Compose([self.knn_graph,GraphKernel(bandwidth)])
    
    def forward(self, graph: Data) -> Data:
        return self.transform(graph)


class KernelRadiusGraph(T.BaseTransform):
    def __init__(self,r: int, bandwidth: float):
        super().__init__()
        self.r = r
        self.bandwidth = bandwidth
        self.radius_graph = T.RadiusGraph(r=self.r,loop=False,num_workers=4)
        self.transform = T.Compose([self.radius_graph,GraphKernel(bandwidth)])
    
    def forward(self, graph: Data) -> Data:
        return self.transform(graph)

class GeometricGraph(Data):
    def __init__(self, edge_index, x, y, edge_weight,pos):
        super().__init__(edge_index=edge_index, edge_attr=edge_weight,x=x,y=y,pos=pos)

    def draw_graph(self):
        from torch_geometric.utils import to_networkx
        import networkx as nx
        import matplotlib.pyplot as plt
        nx_graph = to_networkx(self,to_undirected=True)
        if self.pos is None:
            pos = nx.spring_layout(nx_graph)
        else:
            pos_array = self.pos.numpy()
            pos_array = (pos_array - np.min(pos_array,axis=0))/(np.max(pos_array,axis=0)-np.min(pos_array,axis=0))
            pos_array = (pos_array - np.min(pos_array,axis=0))/(np.max(pos_array,axis=0)-np.min(pos_array,axis=0))
            pos = {i: (p[0], p[1]) for i, p in enumerate(pos_array)}
        nx.draw_networkx_nodes(nx_graph, pos, node_size=10)
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5)
        plt.show()
        

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
    
    def _get_positions(self):
        if self.positions is None:
            return self.positions
        else:
            return torch.DoubleTensor(self.positions)
    
    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                self.positions
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            pos = self._get_positions()
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y,pos=pos, **additional_features)
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

def split_nodes(num_nodes, num_splits,seed=29):
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    splits = []
    for i in range(num_splits):
        nontrain_indices = indices[i::num_splits]  # Disjoint test set for each split
        train_indices = np.setdiff1d(indices, nontrain_indices)
        test_indices = nontrain_indices[:len(nontrain_indices)//2]
        eval_indices = nontrain_indices[len(nontrain_indices)//2:]
        splits.append((train_indices, eval_indices, test_indices))
    return splits