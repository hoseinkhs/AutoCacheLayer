"""
@author: Jun Wang 
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList
import torch
import time
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size, out_h, out_w, cache_enabled=False, return_vectors=False, cache_exits = [], cache_hits = []):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        #CacheUnit 1 here
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        #CacheUnit 2 here
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        #CacheUnit 3 here
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        #self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        #self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(4,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

        self.cache_exits = ModuleList(cache_exits)
        self.cache_hits = cache_hits
        self.shrink_on_hit = True
        self.cache_threshold = None
        self.cache_enabled = cache_enabled
        self.return_vectors = return_vectors
        
        self.layers = [
            self.conv1,
            self.conv2_dw,
            self.conv_23,
            self.conv_3,
            self.conv_34,
            self.conv_4,
            self.conv_45,
            self.conv_5,
            self.conv_6_sep,
            self.conv_6_dw,
            self.conv_6_flatten,
            self.linear,
            self.bn
        ]
        self.cached_layers = [3, 5, 7] #range(len(self.layers))
    
    def forward(self, x):
        results = {"start_time": time.time(), "hit_times": [], "hits":[], "idxs": [torch.arange(x.shape[0])], "outputs": [], "vectors": []}
        idxs = torch.arange(0, x.shape[0])
        cache_active = self.cache_enabled and not self.return_vectors

        def process_exit(out, idxs, exit_idx):
            if not cache_active:
                return out, idxs, False
            cache = self.cache_exits[exit_idx](out)
            hit = self.cache_hits[exit_idx](cache, self.cache_threshold)
            results["hit_times"].append(time.time())
            results["outputs"].append(cache)
            results["hits"].append(hit)
            if self.shrink_on_hit:
                no_hits = torch.logical_not(hit)
                idxs = idxs[no_hits]
                out = out[no_hits]
            results["idxs"].append(idxs)
            return out, idxs, len(idxs) == 0
        exit_idx = 0
        for i in range(len(self.layers)):
            if i in self.cached_layers:
                if self.return_vectors:
                    results["vectors"].append(out)
                if cache_active:
                    out, idxs, should_exit = process_exit(out, idxs, exit_idx)
                    if should_exit:
                        return out, results
                exit_idx += 1
            out = self.layers[i](out if i else x)
        
        return out, results

    def config_cache(self, enabled=None, shrink=None, threshold=None, exits = None, vectors=None, hits=None):
        if enabled is not None:
            self.cache_enabled = enabled
        if vectors is not None:
            self.return_vectors = vectors
        if shrink is not None:
            self.shrink_on_hit = shrink
        if threshold is not None:
            self.cache_threshold = threshold
        if exits is not None:
            self.cache_exits = ModuleList(exits)
        if hits is not None:
            self.cache_hits = hits

