"""
"""

from torch.nn import Linear, Module, ReLU, LogSoftmax, Softmax
import torch

class DenseEmbed(Module):
    def __init__(self, feat_dim, n_l1, n_l2):
        super(DenseEmbed, self).__init__()

        self.block1 = Linear(feat_dim, n_l1)
        self.block2 = Linear(n_l1, n_l2)
        self.relu = ReLU()    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.relu(self.block1(x))
        out = self.relu(self.block2(out))
        return out


class Dense2Layer(Module):
    def __init__(self, feat_dim, n_l1, n_l2):
        super(Dense2Layer, self).__init__()

        self.block1 = Linear(feat_dim, n_l1)
        self.block2 = Linear(n_l1, n_l2)
        self.relu = ReLU()
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.relu(self.block1(x))
        out = self.LogSoftmax(self.block2(out))
        return out

class Dense2LayerSoftmax(Module):
    def __init__(self, feat_dim, n_l1, n_l2):
        super(Dense2LayerSoftmax, self).__init__()

        self.block1 = Linear(feat_dim, n_l1)
        self.block2 = Linear(n_l1, n_l2)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.relu(self.block1(x))
        out = self.softmax(self.block2(out))
        return out


class Dense2LayerTemp(Module):
    def __init__(self, feat_dim, n_l1, n_l2, temp=0.5):
        super(Dense2LayerTemp, self).__init__()
        self.temp = temp
        self.block1 = Linear(feat_dim, n_l1)
        self.block2 = Linear(n_l1, n_l2)
        self.relu = ReLU()
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.relu(self.block1(x))
        out = self.LogSoftmax(self.block2(out)/ self.temp)
        return out 