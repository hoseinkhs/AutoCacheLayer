import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import collections
from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


@model_wrapper
class ModelSpace(nn.Module):
    def __init__(self):
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
        self.fc1 = nn.LazyLinear(nn.ValueChoice([2**i for i in range(5, 10)]))

        self.fc2 = nn.LazyLinear(10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(F.relu(x))
        output = F.log_softmax(self.fc2(x), dim=1)
        return output



def get_model_space():
    return ModelSpace()

# model_space = ModelSpace()

import nni

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)
from nni.retiarii.evaluator import FunctionalEvaluator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
conf_dict = {"train_device": device, "num_classes": 10, 'backbone_type': "Resnet50"}
conf = collections.namedtuple("ConfObject", conf_dict.keys())(*conf_dict.values())
# def train_epoch(cache_model, device, train_loader, optimizer, epoch, backbone):
#     loss_fn = torch.nn.KLDivLoss(log_target=True).cuda(device)
#     cache_model.train()
#     for batch_idx, (samples, label) in enumerate(train_loader):
#         samples, label = samples.to(device), label.to(device)
        
#         optimizer.zero_grad()
#         target, results = backbone(samples, conf, training=True, return_vectors = True)
#         vectors = results.report()['vectors'][0]
#         cache_output = cache_model(vectors)
#         loss = loss_fn(cache_output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(samples), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


# def test_epoch(cache_model, device, test_loader, backbone):
#     cache_model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for samples, target in test_loader:
#             samples, label = samples.to(device), target.to(device)
#             target, results = backbone(samples, conf, training=True, return_vectors = True)
#             vectors = results.report()['vectors'][0]
#             cache_output = cache_model(vectors)
#             # output = model(data)
#             pred = cache_output.argmax(dim=1, keepdim=True)
#             target = target.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)

#     print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#           correct, len(test_loader.dataset), accuracy))

#     return accuracy


# def evaluate_model(model_cls):
#     # "model_cls" is a class, need to instantiate
#     model = model_cls()

    
#     model.to(device)
    

#     backbone_factory = BackboneFactory(
#     "Resnet50", "training_mode/backbone_conf.yaml")
#     model_loader = ModelLoader(backbone_factory)
#     # model = backbone_factory.get_backbone()
#     weights_file = f'models/cifar10/resnet50.pt'

#     backbone = backbone_factory.get_backbone()
#     backbone.load_state_dict(torch.load(weights_file))
#     backbone.eval()
#     backbone.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     transf = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))
#     train_loader = DataLoader(CIFAR10('data/cifar10', download=True, transform=transf), batch_size=64, shuffle=True)
#     test_loader = DataLoader(CIFAR10('data/cifar10', download=True, train=False, transform=transf), batch_size=64)

#     for epoch in range(10):
#         # train the model for one epoch
#         train_epoch(model, device, train_loader, optimizer, epoch, backbone)
#         # test the model for one epoch
#         accuracy = test_epoch(model, device, test_loader, backbone)
#         # call report intermediate result. Result can be float or dict
#         nni.report_intermediate_result(accuracy)

#     # report final test result
#     nni.report_final_result(accuracy)

# evaluate_model(None)
# exit()
# evaluator = FunctionalEvaluator(evaluate_model)

from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
# exp = RetiariiExperiment(get_model_space(), evaluator, [], search_strategy)
# exp_config = RetiariiExeConfig('local')
# exp_config.experiment_name = 'cifar10_search'
# exp_config.max_trial_number =5   # spawn 4 trials at most
# exp_config.trial_concurrency = 2  # will run two trials concurrently
# exp_config.trial_gpu_number = 1
# exp_config.training_service.use_active_gpu = True
# exp.run(exp_config, 8081)
# for model_code in exp.export_top_models(formatter='dict'):
#   print(model_code)
from model_evaluator import get_model_evaluator
backbone_factory = BackboneFactory("Resnet50", "backbone_conf.yaml")

model_loader = ModelLoader(backbone_factory)
        # model = backbone_factory.get_backbone()
weights_file = f'models/cifar10/resnet50.pt'

backbone = backbone_factory.get_backbone()
backbone.load_state_dict(torch.load(weights_file))
backbone.eval()
backbone.to(device)
import os
import yaml

def get_loaders():
    transf = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))
    train_loader = DataLoader(CIFAR10('data/cifar10', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(CIFAR10('data/cifar10', download=True, train=False, transform=transf), batch_size=64)
    return train_loader, test_loader

for num_exit in range(1): #[3, 0, 1, 2]:#
    # pid = os.fork()
    if True: #pid == 0:
        spc = get_model_space()
        exit_layer = num_exit #cached_layers[num_exit]
        evaluator = get_model_evaluator(conf, backbone, num_exit, exit_layer, get_loaders)
        evaluator = FunctionalEvaluator(evaluator)
        exp = RetiariiExperiment(spc, evaluator, [], search_strategy)
        exp_config = RetiariiExeConfig('local')
        exp_config.experiment_name = f'{conf.backbone_type}-exit{num_exit}-layer{exit_layer}'
        exp_config.max_trial_number = 1   # spawn 4 trials at most
        exp_config.trial_concurrency = 2  # will run two trials concurrently
        exp_config.trial_gpu_number = 1
        # exp_config.pickle_size_limit = 32768
        exp_config.training_service.use_active_gpu = True
        # exp_config.execution_engine = 'base'
        # exp_config.export_formatter = 'code'
        exp.run(exp_config, 9081+num_exit)
        print("EXPERIMENT is DONE!")
        for model_dict in exp.export_top_models(formatter='dict'):
            print(model_dict)
            with open(f'cache_model_{num_exit}.yml', 'w') as f:
                yaml.dump(model_dict, outfile, default_flow_style=False)
            break
        break
    else:
        os.wait()
        pass
