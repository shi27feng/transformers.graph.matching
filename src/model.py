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
        self.encoder_ = nn.ModuleList([EncoderLayer(self.gnn_dims[-1],
                                                    args.mha_dim) for _ in range(args.num_enc)])
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

    def enc_pass(self, x, bi):
        for enc in self.encoder_:
            x = enc(x, bi)
        return x

    def forward(self, s, t):
        xs = self.gnn_bone(s.x, s.edge_index, False)
        xt = self.gnn_bone(t.x, t.edge_index, False)
        xs_ = self.enc_pass((xs, xt), (s.batch, t.batch))
        xt_ = self.enc_pass((xt, xs), (t.batch, s.batch))
        return self.fc_(global_mean_pool(xs_, s.batch) +
                        global_mean_pool(xt_, t.batch))
