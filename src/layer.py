from abc import ABC
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import OptPairTensor, Size, OptTensor, Adj
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import matmul, SparseTensor

from attention import LinearAttention


def _norm(edge_index,
          edge_weight=None,
          num_nodes=None,
          improved=False,
          heat_scale=0.,
          add_self_loops=True):
    fill_value = 2. if improved else 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones_like(edge_index.size(1),
                                      dtype=torch.float)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    laplace = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    if heat_scale > 0.:
        return edge_index, torch.exp(-heat_scale * laplace)
    return edge_index, laplace


class GraphConv(MessagePassing, ABC):

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heat_scale=0.,
                 aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heat_scale = heat_scale

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.heat_scale > 0.:
            edge_index, edge_weight = _norm(edge_index,
                                            edge_weight,
                                            add_self_loops=False,
                                            heat_scale=self.heat_scale)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        self.beta = beta
        if self.beta:
            self.lin_beta = nn.Linear(3 * normalized_shape, 1, bias=False)

    def forward(self, x, y):
        if self.beta:
            b = self.lin_beta(torch.cat([y, x, y - x], dim=-1))
            b = b.sigmoid()
            return self.ln(b * x + (1 - b) * self.dropout(y))

        return self.ln(self.dropout(y) + x)


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hd_channels,
                 heads=8,
                 dropout=0.):
        """

        """
        super(EncoderLayer, self).__init__()
        self.in_channels = in_channels
        self.hd_channels = hd_channels
        self.heads = heads
        self.dropout = dropout
        self.lin_kv = Linear(in_channels, hd_channels * 2, bias=False)
        self.lin_q = Linear(in_channels, hd_channels, bias=False)
        self.mha = LinearAttention(in_channels=(hd_channels // heads),
                                   attention_dropout=dropout)

        self.add_norm_att = AddNorm(self.hd_channels, False, self.dropout)
        self.add_norm_ffn = AddNorm(self.hd_channels, False, self.dropout)
        self.ffn = FeedForward(self.hd_channels, self.hd_channels, self.dropout)

    def forward(self, x, bi=None):
        if isinstance(x, torch.Tensor):
            y = w = x
        else:
            y, w = x
        d = y.shape[-1] // self.heads
        query = self.lin_q(y).view(-1, self.heads, d)
        key, value = self.lin_kv(w).chunk(2, dim=-1)

        t = self.mha(query,
                     key.view(-1, self.heads, d),
                     value.view(-1, self.heads, d),
                     bi).view(-1, self.heads * d)

        y = self.add_norm_att(y, t)
        return self.add_norm_ffn(y, self.ffn(y))
