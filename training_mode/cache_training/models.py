import torch
import time

class PlaceModel(torch.nn.Module):
    """Define a classification face model which contains a backbone.
    
    Attributes:
        backbone(object): the backbone of face model.
    """
    def __init__(self, backbone_model):
        """Init face model by backbone factorcy and classifier factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
        """
        super(PlaceModel, self).__init__()
        self.backbone = backbone_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.version = 0

    def forward(self, data, ):
        feat, results = self.backbone.forward(data)
        results["end_time"] = time.time()
        return feat, results

class FaceModel(torch.nn.Module):
    """Define a classification face model which contains a backbone and a classifier.
    
    Attributes:
        backbone(object): the backbone of face model.
        classifier(object): the classifier of face model.
    """
    def __init__(self, backbone_model, classifier_model):
        """Init face model by backbone factorcy and classifier factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            classifier_factory(object): produce a classifier according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_model
        self.classifier = classifier_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.version = 0
        
    def forward(self, data, conf, logger=None,*args, **kwargs):
        feat, ret, results = self.backbone.forward(data, conf, *args, **kwargs)
        if feat.size(0):
            classification = self.classifier.forward(feat)
            results["outputs"].append(classification)
            results["hits"].append(torch.ones(classification.shape[0]))
            ret[results["idxs"][-1]] = classification
            tt = time.time()
            results["hit_times"].append(tt)
            results["end_time"] = tt
            return ret, results
        else:
            results["end_time"] = time.time()
            return ret, results
