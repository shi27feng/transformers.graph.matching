import torch
import torch.nn as nn
import torch.nn.functional as fn
from layer import EncoderLayer
from torch_geometric.nn.conv import GCNConv


class GraphMatchTR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gnn_dims = [args.num_labels] + [int(n) for n in args.gnn_dims.split(',')]
        self.mha_dim = args.mha_dim
        
        self.gnn_ = nn.ModuleList([GCNConv(self.gnn_dims[i], 
                        self.gnn_dims[i + 1]) for i in range(len(self.gnn_dims) - 1)])
        self.encoder = EncoderLayer(self.gnn_dims[-1], args.mha_dim)
        self.fc_ = nn.Sequential(
            nn.Linear(args.mha_dim, args.fc_hidden),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.fc_hidden, 1)
        )

    def gnn_bone(self, h, adj, multi_pass):
        num_layers = len(self.gconv_layers)
        hs = []
        for i in range(num_layers):
            h = self.gnn_[i](h, adj)
            if i is not (num_layers - 1):
                h = fn.dropout(fn.relu(h, inplace=True), p=self.args.dropout, training=self.training)
            hs.append(h.clone())
        return hs if multi_pass else h
    
    def forward(self, s, t):
        s = self.gnn_bone()
        return 
