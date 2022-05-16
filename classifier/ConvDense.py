"""
@author: Amin Abedi
@date: 20211220
@contact: mohammadamin.abedi@ucalgary.ca
"""

from torch.nn import Linear, Module, ReLU, LogSoftmax, Softmax, Conv2d
import torch

class ConvDense(Module):
    def __init__(self, k_l1, in_ch_l1, out_ch_l1, s_l1, in_l2, out_l2):
        super(ConvDense, self).__init__()

        self.conv1 = Conv2d(in_ch_l1, out_ch_l1, k_l1, stride=s_l1)
        self.block2 = Linear(in_l2, out_l2)
        self.relu = ReLU()
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)
        out = self.relu(self.block2(out))
        out = self.LogSoftmax(out)
        return out

class Conv2Dense(Module):
    def __init__(self, k, in_ch, out_ch, s, lin_in, lin_out):
        super(Conv2Dense, self).__init__()

        self.conv1 = Conv2d(in_ch[0], out_ch[0], k[0], stride=s[0])
        self.conv2 = Conv2d(in_ch[1], out_ch[1], k[1], stride=s[1])
        self.block2 = Linear(lin_in, lin_out)
        self.relu = ReLU()
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.block2(out))
        out = self.LogSoftmax(out)
        return out