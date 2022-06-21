"""
@author: Jun Wang 
@date: 20201019 
@contact: jun21wangustc@gmail.com    
"""

import sys
import yaml
sys.path.append('../../')
from backbone.MobileFaceNets import MobileFaceNet
from backbone.EfficientNets import EfficientNet
from backbone.EfficientNets import efficientnet
from backbone.places365.resnet2 import resnet50 as places_resnet50

from backbone.places365.alexnet import places_alexnet
from backbone.places365.densenet import places_densenet

from backbone.cifar10.resnet import resnet18 as cifar10_resnet18, resnet50 as cifar10_resnet50
from backbone.cifar100.resnet import resnet18 as cifar100_resnet18, resnet50 as cifar100_resnet50
class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, backbone_conf_file, experiment=None):
        self.backbone_type = backbone_type
        self.experiment = experiment
        with open(backbone_conf_file) as f:
            backbone_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.backbone_param = backbone_conf[backbone_type]

    def get_backbone(self):
        print(self.backbone_type, self.experiment)
        if self.backbone_type == "Resnet18":
            backbone = cifar10_resnet18() if self.experiment == "Cifar10" else cifar100_resnet18()
        elif self.backbone_type == "Resnet50":
            backbone = cifar10_resnet50() if self.experiment == "Cifar10" else cifar100_resnet50()
        elif self.backbone_type == "PlacesResnet50":
            backbone = places_resnet50()
        elif self.backbone_type == "PlacesAlexNet":
            backbone = places_alexnet()
        elif self.backbone_type == "PlacesDenseNet":
            backbone = places_densenet(pretrained = True, progress=True)
        elif self.backbone_type == 'MobileFaceNet':
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            backbone = MobileFaceNet(feat_dim, out_h, out_w)
        elif self.backbone_type == 'EfficientNet':
            width = self.backbone_param['width'] # width for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            depth = self.backbone_param['depth'] # depth for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            image_size = self.backbone_param['image_size'] # input image size, e.g. 112.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            blocks_args, global_params = efficientnet(
                width_coefficient=width, depth_coefficient=depth, 
                dropout_rate=drop_ratio, image_size=image_size)
            backbone = EfficientNet(out_h, out_w, feat_dim, blocks_args, global_params)
        else:
            pass
        return backbone
