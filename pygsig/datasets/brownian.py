import numpy as np
import pandas as pd
import networkx as nx
import torch
from pygsig.graph import StaticGraphTemporalSignal

class Simulation():
    def __init__( self,
                 num_nodes,
                 num_blocks, 
                 p_across_blocks,
                 p_within_blocks,
                 mu,
                 beta,
                 sigma,
                 omega_noise,
                 time_horizon,
                 task='classification',
                 dt = 1e-3):
        
        self.num_nodes = num_nodes
        self.num_blocks = num_blocks
        self.p_across_blocks = p_across_blocks
        self.p_within_blocks = p_within_blocks
        self.mu = mu
        self.beta = beta
        self.sigma = sigma
        self.omega_noise = omega_noise
        self.time_horizon = time_horizon
        self.dt = dt
        self.num_time_steps = int(time_horizon / dt)
        self.task = task
        self.tt = np.arange(0, self.time_horizon, self.dt)

    def run(self,graph_seed,omega_seed,param_seed):

        # synchronization
        def kuramoto(graph, theta, omega, dt):
            dtheta = omega * dt  # Initialize with intrinsic frequencies
            for u, v, data in graph.edges(data=True):
                coupling = data['weight']
                dtheta[u] += dt * coupling * np.sin(theta[v] - theta[u])
                dtheta[v] += dt * coupling * np.sin(theta[u] - theta[v])
            return theta + dtheta

        # drift of the SDE
        def periodic_drift(beta, theta, omega, mu_0, t):
            return mu_0 + beta*np.sin(omega*t + theta)

        # Create a graph
        block_sizes = [self.num_nodes // self.num_blocks] * self.num_blocks
        block_probs = np.zeros((self.num_blocks, self.num_blocks))

        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if i == j:
                    block_probs[i, j] = self.p_within_blocks
                else:
                    block_probs[i, j] = self.p_across_blocks
        
        graph = nx.stochastic_block_model(block_sizes, block_probs, seed=graph_seed)

        for edge in graph.edges:
            if graph.nodes[edge[0]]['block'] == graph.nodes[edge[1]]['block']:
                graph[edge[0]][edge[1]]['weight'] = 1/(np.sqrt(graph.degree[edge[0]]*graph.degree[edge[1]]))
            else:
                graph[edge[0]][edge[1]]['weight'] = 1/(np.sqrt(graph.degree[edge[0]]*graph.degree[edge[1]]))
        
        # Assign omega to each node
        np.random.seed(omega_seed)
        omega_range = np.linspace(0,1, self.num_blocks+1)[1:]
        for node in graph.nodes:
            graph.nodes[node]['omega'] = max(omega_range[graph.nodes[node]['block']] + self.omega_noise * np.random.randn(),1e-3)
        
        # Othe oscilator perameters
        np.random.seed(param_seed)
        omega = np.array([graph.nodes[node]['omega'] for node in graph.nodes])
        block = np.array([graph.nodes[node]['block'] for node in graph.nodes])
        beta =  self.beta * np.ones(self.num_nodes) # amplitude (uniform nodes)
        theta = 2 * np.pi * np.random.rand(self.num_nodes)  # initial phase (random across nodes)
        mu_0 = self.mu * np.ones(self.num_nodes)

        # initial values
        X = np.random.rand(self.num_nodes) # signal

        # Simulate
        theta_traj = np.zeros((self.num_nodes,self.num_time_steps))
        mu_traj = np.zeros((self.num_nodes,self.num_time_steps))
        X_traj = np.zeros((self.num_nodes,self.num_time_steps))

        # Time sequence
        tt = np.arange(0, self.time_horizon, self.dt)
        for step,t in enumerate(tt):
            theta_traj[:, step] = theta
            if step == 0:
                mu_traj[:,step] = mu_0
            else:
                mu_traj[:,step] = mu
            X_traj[:,step] = X
            theta = kuramoto(graph, theta, omega,self.dt)
            mu = periodic_drift(beta, theta,omega, mu_0, t)
            X = X + self.dt * mu + np.sqrt(self.dt) * self.sigma * np.random.randn(self.num_nodes)
        
        self.X = X_traj
        self.theta = theta_traj
        self.block = block
        self.graph = graph
        self.omega = omega
    
    def get_sequence(self):
        from sklearn.preprocessing import OneHotEncoder
        one_hot = OneHotEncoder()

        if self.task == 'classification':
            y = one_hot.fit_transform(self.block.reshape(-1,1)).toarray()
        if self.task == 'regression':
            y = self.omega.reshape(-1,1)

        snapshot_count = self.X.shape[1]
        df_edge = nx.to_pandas_edgelist(self.graph.to_directed())
        edge_index = torch.tensor(df_edge[['source','target']].values.T,dtype=torch.long)
        edge_weight = torch.tensor(df_edge['weight'].values,dtype=torch.float)
        snapshot_count = self.X.shape[1]
        features = [ torch.tensor(self.X[:,t],dtype=torch.float).unsqueeze(-1) for t in range(snapshot_count)]
        targets = [ y for _ in range(snapshot_count)]
        # Sequential Data
        return StaticGraphTemporalSignal(edge_index=edge_index,edge_weight=edge_weight,features=features,targets=targets)
