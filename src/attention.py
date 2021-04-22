import torch
import torch.nn as nn
from fast_transformers.feature_maps import elu_feature_map


class LinearAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 softmax_temp=None,
                 feature_map=None,
                 eps=1e-6,
                 attention_dropout=0.1):
        super(LinearAttention, self).__init__()
        self.in_channels = in_channels
        self.softmax_temp = softmax_temp
        self.dropout = attention_dropout
        self.eps = eps
        self.feature_map = (
            feature_map(in_channels) if feature_map else
            elu_feature_map(query_dims=in_channels)
        )

    def forward(self, queries, keys, values):
        n, l, h, e = queries.shape  # batch, length, heads, depth
        _, _, s, d = values.shape
        assert keys.shape[1] == values.shape[1], "key's and value's length are not matching"
        softmax_temp = self.softmax_temp or (e ** -0.25)  # TODO: how to use this?
        (queries, keys) = map(lambda x: x * softmax_temp, (queries, keys))
        self.feature_map.new_feature_map(queries.device)
        q = self.feature_map.forward_queries(queries)
        k = self.feature_map.forward_keys(keys)

        kv = torch.einsum("nshd, nshm -> nhmd", k, values)
        z = 1 / (torch.einsum("nlhd, nhd -> nlh", q, k.sum(dim=1)) + self.eps)
        v = torch.einsum("nlhd, nhmd, nlh -> nlhm", q, kv, z)

        return v.contiguous()
