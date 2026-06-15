import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import torch as t
from scipy.sparse.linalg import lobpcg
from params import args
import warnings

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='scipy.sparse.linalg._lobpcg'
)
warnings.filterwarnings(
    'ignore',
    message=r'Exited (at iteration|postprocessing).*',
    category=UserWarning
)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class HiESEncoder(nn.Module):
    
    def __init__(self, dim, feat_name='lap_pos_enc'):
        super().__init__()
        self.hidden_dim = dim
        self.k_eigen = args.k_eigen  
        self.feat_name = feat_name

        
        self.mlp_lsb = nn.Sequential(
            nn.Linear(5, dim),  # [deg, mean, std, max, min]
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )

       
        self.mlp_lpb = nn.Sequential(
            nn.Linear(self.k_eigen, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )

        
        self.gin_layers = nn.ModuleList([
            nn.Linear(dim, dim),
            nn.Linear(dim, dim)
        ])
        self.eps = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(2)])

       
        self.gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        self.to(device)

    def compute_local_statistics(self, adj):
        
        num_nodes = adj.shape[0]

       
        deg = np.array(adj.sum(axis=1)).flatten()
        mean = np.zeros(num_nodes)
        std = np.zeros(num_nodes)
        max_ = np.zeros(num_nodes)
        min_ = np.zeros(num_nodes)

        for v in range(num_nodes):
            neighbors = adj[v].indices  
            if len(neighbors) == 0:
                mean[v] = 0
                std[v] = 0
                max_[v] = 0
                min_[v] = 0
            else:
                nd = deg[neighbors]
                mean[v] = np.mean(nd)
                std[v] = np.std(nd)
                max_[v] = np.max(nd)
                min_[v] = np.min(nd)

        
        stats = np.stack([deg, mean, std, max_, min_], axis=1)
        return torch.tensor(stats, dtype=torch.float32)

    def compute_spectral_embedding(self, adj):
        
        num_nodes = adj.shape[0]
        
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0  
        
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg), format='csr')
        
        I = sp.eye(num_nodes, format='csr')  
        L_sym = I - D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

        X = np.random.rand(num_nodes, self.k_eigen)
        eigvals, eigvecs = lobpcg(
            L_sym,
            X,
            largest=False,
            maxiter=200,  
            tol=1e-1  
        )

        
        eigvals, eigvecs = eigsh(
            L_sym,
            k=self.k_eigen,
            which='SM',  
            maxiter=200,
            tol=1e-3
        )

       
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)
        return torch.tensor(eigvecs, dtype=torch.float32)

    def forward(self, stats, spectral, g):
        
        h_local = self.mlp_lsb(stats)
        
        h_spectral = self.mlp_lpb(spectral)

        
        x = h_local + h_spectral
        
        for l in range(2):
            
            g.ndata['h'] = x
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata.pop('agg')  

           
            x = self.gin_layers[l]((1 + self.eps[l]) * x + agg)

            if l != 1:
                x = F.leaky_relu(x)
        h_learned = x

        
        gate_input = torch.cat([h_local, h_spectral, h_learned], dim=1)
        gv = self.gate(gate_input)
        fused_part = self.fusion_mlp(torch.cat([h_spectral, h_learned], dim=1))
        c_r = gv * h_local + (1 - gv) * fused_part


        
        g.ndata[self.feat_name] = c_r
        return c_r
