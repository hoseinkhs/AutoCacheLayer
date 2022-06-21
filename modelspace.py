import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import collections
from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory
from functools import partial

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Conv2dBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.LazyConv2d(
            *args,
            **kwargs
        )

    def forward(self, x):
        return F.relu(self.conv(x))

class LinearBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.LazyLinear(*args, **kwargs)

    def forward(self, x):
        return F.relu(self.linear(x))

@model_wrapper
class ModelSpace(nn.Module):
    
    def __init__(self, num_classes = 10):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        # self.conv2 = nn.LayerChoice([
        #     nn.Conv2d(32, 64, 3, 1),
        #     DepthwiseSeparableConv(32, 64)
        # ])
        # ValueChoice is used to select a dropout rate.
        # ValueChoice can be used as parameter of modules wrapped in `nni.retiarii.nn.pytorch`
        # or customized modules wrapped with `@basic_unit`.
        # self.dropout1 = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        # # self.dropout2 = nn.Dropout(0.5)
        # self.conv_stack = nn.Repeat(lambda index: Conv2dBlock(
        #     nn.ValueChoice([2**i for i in range(5, 7)]),
        #     nn.ValueChoice([1, 3, 5]),
        #     nn.ValueChoice(range(1, 4))) if index else nn.Identity(), nn.ValueChoice(range(1, 4)))
        # self.linear_stack = nn.Repeat(lambda index: LinearBlock(nn.ValueChoice([2**i for i in range(5, 12)])) if index else nn.Identity(), 3)
        self.conv1 = nn.LayerChoice([nn.Identity(), Conv2dBlock(nn.ValueChoice([2**i for i in range(5, 7)]),
            nn.ValueChoice([1, 3, 5]),
            nn.ValueChoice(range(1, 4)))])
        self.conv2 = nn.LayerChoice([nn.Identity(), Conv2dBlock(nn.ValueChoice([2**i for i in range(5, 7)]),
            nn.ValueChoice([1, 3, 5]),
            nn.ValueChoice(range(1, 4)))])
        self.conv3 = nn.LayerChoice([nn.Identity(), Conv2dBlock(nn.ValueChoice([2**i for i in range(5, 7)]),
            nn.ValueChoice([1, 3, 5]),
            nn.ValueChoice(range(1, 4)))])
        self.lin1 = nn.LayerChoice([nn.Identity(), LinearBlock(nn.ValueChoice([2**i for i in range(5, 12)]))])
        self.lin2 = nn.LayerChoice([nn.Identity(), LinearBlock(nn.ValueChoice([2**i for i in range(5, 12)]))])
        self.lin3 = nn.LayerChoice([nn.Identity(), LinearBlock(nn.ValueChoice([2**i for i in range(5, 12)]))])
        # self.l1 = nn.LayerChoice([
        #     nn.LazyLinear(nn.ValueChoice([2**i for i in range(5, 12)])),
        #     nn.Identity()
        # ])
        # self.fc1 = nn.LazyLinear(nn.ValueChoice([2**i for i in range(5, 10)]))

        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.lin3(self.lin2(self.lin1(x)))
        output = F.log_softmax(self.classifier(x), dim=1)
        return output



def get_model_space():
    return ModelSpace()


import nni
import os
import yaml

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from model_evaluator import evaluate_model

search_strategy = strategy.Random(dedup=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Experiment():
    def __init__(self, conf=None):
        conf_dict = {"train_device": device, "train_epochs": 10, "num_classes": 10, 'backbone_type': "Resnet50", "lr":0.1, "momentum": 0.9, "milestones": [10, 13, 16]}
        self.conf = collections.namedtuple("ConfObject", conf_dict.keys())(*conf_dict.values())

    def get_backbone(self):
        backbone_factory = BackboneFactory("Resnet50", "backbone_conf.yaml", experiment="Cifar10")
        model_loader = ModelLoader(backbone_factory)
        weights_file = f'models/cifar10/resnet50.pt'
        backbone = backbone_factory.get_backbone()
        backbone.load_state_dict(torch.load(weights_file))
        return backbone

    def get_loaders(self):
        transf = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))
        test_loader = DataLoader(CIFAR10('data/cifar10', download=True, transform=transf), batch_size=64, shuffle=True)
        train_loader = DataLoader(CIFAR10('data/cifar10', download=True, train=False, transform=transf), batch_size=64)
        return train_loader, test_loader
        
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

exp = Experiment()
# for num_exit in range(1):
#     pid = os.fork()
#     if pid == 0:
spc = get_model_space()
num_exit = 0
evaluator = partial(evaluate_model, exp, num_exit, device)
evaluator = FunctionalEvaluator(evaluator)
ret = RetiariiExperiment(spc, evaluator, [], search_strategy)
ret_config = RetiariiExeConfig('local')
ret_config.experiment_name = f'{exp.conf.backbone_type}-exit{num_exit}'
ret_config.max_trial_number = 1   # spawn 4 trials at most
ret_config.trial_concurrency = 2  # will run two trials concurrently
ret_config.trial_gpu_number = 1
# ret_config.pickle_size_limit = 65536
ret_config.training_service.use_active_gpu = True
# ret_config.execution_engine = 'base'
# ret_config.export_formatter = 'code'
ret.run(ret_config, 8080+num_exit)
print("EXPERIMENT is DONE!")
for model_dict in ret.export_top_models(formatter='dict'):
    print(model_dict)
    with open(f'cache_model_{num_exit}.yml', 'w') as f:
        yaml.dump(model_dict, outfile, default_flow_style=False)
    break
# break
#     else:
#         os.wait()
#         pass
