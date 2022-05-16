"""
@author: Amin Abedi
@date: 20211220
@contact: mohammadamin.abedi@ucalgary.ca
"""

from torch.nn import Linear, Module, ReLU, LogSoftmax, Softmax, Conv2d, ModuleList
import torch

class Attention(Module):
    def __init__(self, in_channels, in_features, out_features, N=1):
        super(Attention, self).__init__()
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride= 2, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride= 1)
        self.depthwises = ModuleList([Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride= 2, groups=in_channels) for i in range(N)])
        self.pointwises = ModuleList([Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride= 1) for i in range(N)])
        self.linear = Linear(in_features, out_features)
        self.LogSoftmax = LogSoftmax(dim=1)
    
    def forward(self, out):
        # att = self.depth_conv(x)
        # att = self.point_conv(att)
        # att = torch.nn.functional.interpolate(att, mode="bilinear")
        # out = torch.dot(x, att)
        for depth, point in zip(self.depthwises, self.pointwises):
            out = depth(out)
            out = point(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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