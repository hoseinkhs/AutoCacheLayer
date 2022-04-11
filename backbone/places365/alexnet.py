import torch
import torch.nn as nn
import os
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList, LogSoftmax, Softmax
from PIL import Image
import time
import pathlib

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, cache_enabled=False, return_vectors=False, cache_exits = [], cache_hits = []) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.layers = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), #0
            nn.ReLU(inplace=True), #1
            nn.MaxPool2d(kernel_size=3, stride=2), #2
            nn.Conv2d(64, 192, kernel_size=5, padding=2), #3
            nn.ReLU(inplace=True),#4
            nn.MaxPool2d(kernel_size=3, stride=2),#5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),#6
            nn.ReLU(inplace=True),#7
            nn.Conv2d(384, 256, kernel_size=3, padding=1),#8
            nn.ReLU(inplace=True),#9
            nn.Conv2d(256, 256, kernel_size=3, padding=1),#10
            nn.ReLU(inplace=True),#11
            nn.MaxPool2d(kernel_size=3, stride=2),#12
            nn.AdaptiveAvgPool2d((6, 6)),#13
            nn.Dropout(p=dropout),#14
            nn.Linear(256 * 6 * 6, 4096),#15
            nn.ReLU(inplace=True),#16
            nn.Dropout(p=dropout),#17
            nn.Linear(4096, 4096),#18
            nn.ReLU(inplace=True),#19
            nn.Linear(4096, num_classes),#20
            nn.LogSoftmax()#21
            ]
        self.features = nn.Sequential(*self.layers[:13])
        self.avgpool = self.layers[13] 
        self.classifier = nn.Sequential(*self.layers[14:])

        self.cache_exits = ModuleList(cache_exits)
        self.cache_hits = cache_hits
        self.shrink_on_hit = True
        self.cache_threshold = None
        self.cache_enabled = cache_enabled
        self.return_vectors = return_vectors

        self.cached_layers = [7, 9, 11] 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x
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
            if i == 14:
                out = out.view(out.size(0), -1)
            out = self.layers[i](out if i else x)
            if i in self.cached_layers:
                if self.return_vectors:
                    results["vectors"].append(out)
                if cache_active:
                    out, idxs, should_exit = process_exit(out, idxs, exit_idx)
                    if should_exit:
                        return out, results
                exit_idx += 1
            
        results["outputs"].append(out)
        results["hits"].append(torch.ones(out.shape[0]))
        results["hit_times"].append(time.time())
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

def places_alexnet(arch = 'alexnet', **kwargs):
    # load the pre-trained weights
    model_file = os.path.join(pathlib.Path(__file__).parent.resolve(),'%s_places365.pth.tar' % arch)
    if not os.access(model_file, os.W_OK):
        print(model_file)
        raise Exception("FILE NOT FOUND")
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = AlexNet(num_classes=365, **kwargs)# models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model
