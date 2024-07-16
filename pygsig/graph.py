import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
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
    def __init__(self, edge_index, edge_attr, pos, x=None, y=None, **kwargs):
        super().__init__(self, edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, pos=pos, **kwargs)

    def draw_graph(self):
        from torch_geometric.utils import to_networkx
        import networkx as nx
        import matplotlib.pyplot as plt
        pos_array = self.pos.numpy()
        pos_array = (pos_array - np.min(pos_array,axis=0))/(np.max(pos_array,axis=0)-np.min(pos_array,axis=0))
        pos_array = (pos_array - np.min(pos_array,axis=0))/(np.max(pos_array,axis=0)-np.min(pos_array,axis=0))
        pos = {i: (p[0], p[1]) for i, p in enumerate(pos_array)}
        nx_graph = to_networkx(self,to_undirected=True)
        nx.draw_networkx_nodes(nx_graph, pos, node_size=10)
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5)
        plt.show()
        

class CustomStaticGraphTemporalSignal(StaticGraphTemporalSignal):
    def __init__(self,
                edge_index: Edge_Index,
                edge_weight: Edge_Weight,
                features: Node_Features,
                targets: Targets,
                positions: Positions,
                **kwargs: Additional_Features
    ):
        super().__init__(edge_index, edge_weight, features, targets)
        # self.graph = GeometricGraph(edge_index=edge_index,edge_attr=edge_weight,pos= positions)
        self.positions = positions
        self.num_nodes = self.features[0].shape[0]
        self.num_node_features = self.features[0].shape[-1]
        # self.y = super()._get_target(-1)
    
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

    def plot_graph(self):
        import networkx as nx
        import plotly.graph_objs as go
        import numpy as np

        
class RandomNodeSplit(T.BaseTransform):
    def __init__(self,train_ratio,eval_ratio=0.0,num_splits=1,unlabeled_data=False,class_weights=False,seed=None):
        self.num_splits = num_splits
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.unlabeled_data = unlabeled_data
        self.class_weights = class_weights
        self.seed = seed
        self.base_generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
    
    def _get_classes(self,y):
        classes = torch.unique(y).numpy()
        return np.delete(classes,np.where(classes == -1))

    def forward(self, data: Union[Data, StaticGraphTemporalSignal]):
        num_labeled_nodes = (data.y != -1).sum().item()
        self.train_size = int(self.train_ratio * num_labeled_nodes)
        self.eval_size = int(self.eval_ratio * num_labeled_nodes)
        self.test_size = num_labeled_nodes - self.train_size - self.eval_size

        train_mask = torch.zeros([self.num_splits, data.num_nodes], dtype=torch.bool)
        eval_mask = torch.zeros([self.num_splits, data.num_nodes], dtype=torch.bool)
        test_mask = torch.zeros([self.num_splits, data.num_nodes], dtype=torch.bool)
        label_mask = (data.y != -1)

        labeled_indices = torch.where(label_mask)[0]
        if self.base_generator is not None:
            perm = torch.stack([torch.randperm(labeled_indices.size(0),generator=self.base_generator.manual_seed(self.seed+i)) for i in range(self.num_splits)])
        else:
            perm = torch.stack([torch.randperm(labeled_indices.size(0)) for _ in range(self.num_splits)])

        for i in range(self.num_splits):
            train_mask[i, labeled_indices[perm[i, :self.train_size]]] = True
            eval_mask[i, labeled_indices[perm[i, self.train_size:self.train_size + self.eval_size]]] = True
            test_mask[i, labeled_indices[perm[i, self.train_size + self.eval_size:]]] = True
        
        # class weights
        if self.class_weights and self.unlabeled_data:
            classes = self._get_classes(data.y)
            class_weights = torch.zeros([self.num_splits,len(classes)])
            for i in range(self.num_splits):
                class_weights[i] = torch.Tensor(compute_class_weight(class_weight='balanced',classes=classes,y=data.y[train_mask[i]].numpy()))

            if isinstance(data, Data):
                return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, pos=data.pos,
                            num_splits=self.num_splits,class_weights=class_weights,
                            label_mask=label_mask, train_mask=train_mask, eval_mask=eval_mask, test_mask=test_mask)
            if isinstance(data, CustomStaticGraphTemporalSignal):
                return CustomStaticGraphTemporalSignal(edge_index=data.edge_index, edge_weight=data.edge_weight, features=data.features, targets=data.targets, positions=data.positions,
                                                num_splits=self.num_splits,class_weights=[class_weights]*data.snapshot_count,label_masks=[label_mask]*data.snapshot_count, 
                                                train_masks=[train_mask]*data.snapshot_count,eval_masks=[eval_mask]*data.snapshot_count, test_masks=[test_mask]*data.snapshot_count)
        else:
            if isinstance(data, Data):
                return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, pos=data.pos,
                            num_splits=self.num_splits, train_mask=train_mask, eval_mask=eval_mask, test_mask=test_mask)
            if isinstance(data, CustomStaticGraphTemporalSignal):
                return CustomStaticGraphTemporalSignal(edge_index=data.edge_index, edge_weight=data.edge_weight, features=data.features, targets=data.targets, positions=data.positions,
                                                num_splits=self.num_splits, 
                                                train_masks=[train_mask]*data.snapshot_count,eval_masks=[eval_mask]*data.snapshot_count, test_masks=[test_mask]*data.snapshot_count)


class RandomNodeCV(T.BaseTransform):
    def __init__(self,num_splits=1,unlabeled_data=False,seed=None):
        pass

    def forward(self, data: Union[Data, StaticGraphTemporalSignal]):
        pass