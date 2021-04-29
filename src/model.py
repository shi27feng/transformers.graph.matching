import torch.nn as nn
import torch.nn.functional as fn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool

from layer import EncoderLayer


class GraphMatchTR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gnn_dims = [args.num_labels] + [int(n) for n in args.gnn_dims.split(',')]
        self.mha_dim = args.mha_dim

        self.gnn_ = nn.ModuleList([GCNConv(self.gnn_dims[i],
                                           self.gnn_dims[i + 1]) for i in range(len(self.gnn_dims) - 1)])
        self.encoder_ = nn.ModuleList([EncoderLayer(self.gnn_dims[i + 1], self.gnn_dims[i + 1])
                                       for i in range(len(self.gnn_dims) - 1)])
        self.fc_ = nn.Sequential(
            nn.Linear(args.mha_dim, args.fc_hidden),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.fc_hidden, 1)
        )

    def forward(self, s, t):
        num_layers = len(self.gnn_)
        xs, xt = s.x, t.x
        adj_s, adj_t = s.edge_index, t.edge_index
        for i in range(num_layers):
            xs_ = self.gnn_[i](xs, adj_s)
            xt_ = self.gnn_[i](xt, adj_t)
            if i is not (num_layers - 1):
                xs_ = fn.dropout(fn.relu(xs_, inplace=True), p=self.args.dropout, training=self.training)
                xt_ = fn.dropout(fn.relu(xt_, inplace=True), p=self.args.dropout, training=self.training)

            xs = self.encoder_[i]((xs_, xt_), (s.batch, t.batch))
            xt = self.encoder_[i]((xt_, xs_), (t.batch, s.batch))

        return self.fc_(global_mean_pool(xs, s.batch) +
                        global_mean_pool(xt, t.batch)).squeeze(-1)
