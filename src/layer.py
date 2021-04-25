import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from attention import LinearAttention
from einops import rearrange


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, beta, dropout, heads, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
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

        self.add_norm_att = AddNorm(self.hd_channels,
                                    False, self.dropout, self.heads)
        self.add_norm_ffn = AddNorm(self.hd_channels,
                                    False, self.dropout, self.heads)
        self.ffn = FeedForward(self.hd_channels,
                               self.hd_channels, self.dropout)

    def forward(self, x, bi=None):
        if isinstance(x, torch.Tensor):
            y = w = x
        else:
            y, w = x
        d = y.shape[-1] // self.heads
        query = self.lin_q(y).view(-1, d, self.heads)
        key, value = self.lin_kv(w).chunk(2, dim=-1)

        t = self.mha(query,
                     key.view(-1, d, self.heads),
                     value.view(-1, d, self.heads),
                     bi).view(-1, d * self.heads)

        y = self.add_norm_att(y, t)
        return self.add_norm_ffn(y, self.ffn(y))
