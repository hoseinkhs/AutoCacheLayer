import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList, LogSoftmax, Softmax
from PIL import Image
import time
import pathlib
import sys
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, cache_enabled=False, return_vectors=False, cache_exits = [], cache_hits = []):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.log_softmax = LogSoftmax(dim = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.cache_exits = ModuleList(cache_exits)
        self.cache_hits = cache_hits
        self.shrink_on_hit = True
        self.cache_threshold = None
        self.cache_enabled = cache_enabled
        self.return_vectors = return_vectors
        
        self.layers = [
        self.conv1,
        self.bn1,
        self.relu,
        #self.maxpool,

        self.layer1,
        self.layer2,
        self.layer3,
        self.layer4,

        self.avgpool,
        #x.view(x.size(0), ,
        self.fc,
        self.log_softmax
        ]
        self.cached_layers = [3] #range(len(self.layers))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # #x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
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
            if i == len(self.layers)-2:
                out = out.view(out.size(0), -1)
            out = self.layers[i](out if i else x)
        results["outputs"].append(out)
        results["hits"].append(torch.ones(out.shape[0]))
        results["hit_times"].append(time.time())
        return out, results

        # return x
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

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model




# th architecture to use
def places_resnet50(arch = 'resnet50', **kwargs):
    
    # load the pre-trained weights
    
    model_file = os.path.join(pathlib.Path(__file__).parent.resolve(),'%s_places365.pth.tar' % arch)
    if not os.access(model_file, os.W_OK):
        print(model_file)
        raise Exception("FILE NOT FOUND")
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = resnet50(num_classes=365, **kwargs)# models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

# model = places_resnet50()
# model.eval()
# # load the image transformer
# centre_crop = trn.Compose([
#         trn.Resize((256,256)),
#         trn.CenterCrop(224),
#         trn.ToTensor(),
#         trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # load the class label
# file_name = 'categories_places365.txt'
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
# classes = list()
# with open(file_name) as class_file:
#     for line in class_file:
#         classes.append(line.strip().split(' ')[0][3:])
# classes = tuple(classes)

# # load the test image
# img_name = '12.jpg'
# if not os.access(img_name, os.W_OK):
#     img_url = 'http://places.csail.mit.edu/demo/' + img_name
#     os.system('wget ' + img_url)

# img = Image.open(img_name)
# input_img = V(centre_crop(img).unsqueeze(0))

# # forward pass
# logit, _ = model.forward(input_img)
# h_x = F.softmax(logit, 1).data.squeeze()
# probs, idx = h_x.sort(0, True)

# print('{} prediction on {}'.format("resnet50",img_name))
# # output the prediction
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

