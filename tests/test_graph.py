import pytest
import sys

sys.path.append('../pygsig')

import torch
from pygsig.graph import CustomStaticGraphTemporalSignal

def custom_static_graph_temporal_signal():
    # Create graph
    num_nodes = 100
    num_edges = 40
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges)
    # Create features / positions
    snapshot_count = 10
    num_node_features = 5
    features = [torch.randn(num_nodes, num_node_features) for _ in range(snapshot_count)]
    targets = [torch.randint(0,2,(num_nodes,),dtype=torch.float) for _ in range(snapshot_count)]
    pos_dim = 2
    positions = torch.rand(num_nodes, pos_dim,dtype=torch.double)

    dataset = CustomStaticGraphTemporalSignal(edge_index,edge_weight,features,targets,positions)
    assert dataset[0].x.shape == (num_nodes,num_node_features)
    assert dataset[0].y.shape == (num_nodes,)
    assert dataset.snapshot_count == snapshot_count
    assert dataset[0].edge_index.shape == (2,num_edges)
    assert dataset[0].edge_attr.shape == (num_edges,)
