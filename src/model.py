import torch
import torch.nn as nn
from layer import EncoderLayer


class GraphMatchTR(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
