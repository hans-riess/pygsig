import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from pygsig.graph import StaticGraphTemporalSignal,GeometricGraph
from signatory import LogSignature,Signature
from tslearn.metrics import dtw
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import pickle


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

class TS2VecFeatures(T.BaseTransform):
    def __init__(self, encoder, normalize=False):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize

    def forward(self, dataset):
        X = torch.stack([snapshot.x for snapshot in dataset]).transpose(0, 1).numpy()
        x = torch.tensor(self.encoder.encode(X, encoding_window='full_series'))
        y = dataset[-1].y
        pos = dataset[-1].pos
    
        # Normalize if required
        if self.normalize:
            std_x = torch.std(x, dim=0)
            mean_x = torch.mean(x, dim=0)
            x = (x - mean_x) / std_x

        # Create the static graph dataset with transformed features
        dataset_static = Data(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static

class RandomFeatures(T.BaseTransform):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, dataset):
        x = torch.randn(dataset.num_nodes,self.num_features)
        y = dataset[-1].y
        pos = dataset[-1].pos

        # Create the static graph dataset with transformed features
        dataset_static = Data(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static

class DTWFeatures(T.BaseTransform):
    def __init__(self,num_features,normalize=False,dtw_path=None):
        super().__init__()
        self.dtw_path = dtw_path
        self.num_features = num_features
        self.normalize = normalize
        if dtw_path is not None:
            self.DTW = torch.load(dtw_path).numpy()
    
    def forward(self, dataset):
        X = torch.stack([snapshot.x for snapshot in dataset]).transpose(0, 1).numpy()
        y = dataset[-1].y
        pos = dataset[-1].pos
        if self.DTW is None:
            DTW = np.zeros(dataset.num_nodes,dataset.num_nodes)
            for i in range(dataset.num_nodes):
                for j in range(dataset.num_nodes):
                    if i < j:
                        DTW[i,j] = dtw(X[i,:,:], X[j,:,:])
                        DTW[j,i] = DTW[i,j]
            self.DTW = DTW
   
        # Apply multidimensional scaling
        mds = MDS(n_components=self.num_features, dissimilarity='precomputed',normalized_stress='auto')
        x = torch.tensor(mds.fit_transform(self.DTW)).float()

        # Normalize if required
        if self.normalize:
            std_x = torch.std(x, dim=0)
            mean_x = torch.mean(x, dim=0)
            x = (x - mean_x) / std_x

        # Create the static graph dataset with transformed features
        dataset_static = Data(x=x, y=y, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight, pos=pos)
        return dataset_static