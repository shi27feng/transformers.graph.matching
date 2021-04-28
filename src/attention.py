import torch
import torch.nn as nn
from fast_transformers.feature_maps import elu_feature_map
from torch_scatter import scatter_sum


class LinearAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 feature_map=None,
                 eps=1e-6,
                 attention_dropout=0.1):
        super(LinearAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout
        self.eps = eps
        self.feature_map = (
            feature_map(in_channels) if feature_map else
            elu_feature_map(query_dims=in_channels)
        )

    def forward(self, queries, keys, values, bi=None):
        e = queries.shape[-1]  # batch, n_heads, length, depth
        softmax_temp = self.softmax_temp or (e ** -0.25)
        (queries, keys) = map(lambda x: x * softmax_temp, (queries, keys))
        self.feature_map.new_feature_map(queries.device)
        q = self.feature_map.forward_queries(queries)
        k = self.feature_map.forward_keys(keys)

        if bi is None:
            kv = torch.einsum("nshd, nshm -> nhmd", k, values)
            z = 1 / (torch.einsum("nlhd, nhd -> nlh", q, k.sum(dim=1)) + self.eps)
            v = torch.einsum("nlhd, nhmd, nlh -> nlhm", q, kv, z)
        else:
            if isinstance(bi, torch.Tensor):
                biq = biv = bi
            else:
                biq, biv = bi 
            q = q.transpose(-3, -2)
            k = k.transpose(-3, -2)
            values = values.transpose(-3, -2)
            # change the dimensions of keys to (...,H, L, D, 1) and values to (..., H, L, 1, D)
            kv = torch.matmul(k.unsqueeze(-1), values.unsqueeze(-2))  # ...HL(D1) \times ...HL(1D) -> ...HL(DD)
            kv = scatter_sum(kv, biv, dim=-3).index_select(dim=-3, index=biq)  # ...H(L)DD
            k_ = scatter_sum(k, biv, dim=-2).index_select(dim=-2, index=biq)  # ...H(L)D
            z = 1 / torch.sum(q * k_, dim=-1)
            v = torch.matmul(q.unsqueeze(-2), kv).squeeze(dim=-2) * z.unsqueeze(-1)  # ...HL(1D) \times ...HL(DD)
        return v.transpose(-3, -2).contiguous()
