import torch
import torch.nn as nn
import os
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList, LogSoftmax, Softmax
from PIL import Image
from backbone.CacheControl import CacheControl
import time

import pathlib

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
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

        self.cached_layers = [7, 9, 11] 

    def forward(self, x: torch.Tensor, args=None, cache=False, return_vectors=False, threshold = 1, training=False, logger=None) -> torch.Tensor:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x
        if args:
            cc = CacheControl(args, x.shape, threshold, self.cache_exits, training, logger = logger)
        for i in range(len(self.layers)):
            if args and i in self.cached_layers:
                if return_vectors:
                    cc.vectors.append(out)
                if cache:
                    if logger:
                        logger.info("CHECKING CACHE")
                    out, should_exit = cc.exit(out)
                    if should_exit:
                        return out, cc.ret, cc.report()
            if i == 14:
                out = torch.flatten(out, 1)
            out = self.layers[i](out if i else x)
        if args:
            cc.exit(out, final=True)
            return out, cc.ret, cc.report()
        else:
            return out

        # return self._forward_impl(x)

    def set_exit_models(self, models):
        self.cache_exits = nn.ModuleList(models)


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
