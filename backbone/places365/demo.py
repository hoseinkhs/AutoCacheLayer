import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import numpy as np
import sys
sys.path.append('../../')
from torch.utils.data import DataLoader
from data_processor.train_dataset import ImageDataset, PlaceDataset, transform as aug_tf

# th architecture to use
arch = 'resnet50'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
model = model.to('cuda:0')

def top_k_categorical_accuracy(y_true, y_pred_proba, k=1):
    # print(y_true[:, None].shape)
    # print(np.argsort(y_pred_proba))
    return 

print(model)
test_data = PlaceDataset("../../data/places/places365_standard","../../data/places/places365_standard/val.txt", "../../data/places/places365_standard/names.txt")
test_data_loader = DataLoader(test_data, 
                             16, True, num_workers = 0)
total = 0
correct = 0 
correct5 = 0 
for images, labels in test_data_loader:
    total += labels.size(0)
    # labels = labels.unsqueeze(1)
    images = images.to('cuda:0')
    out = model.forward(images)
    out = out.to('cpu').detach()
    h_x = F.softmax(out, 1).data.squeeze()
    _, predicted = torch.max(out, 1)
    probs, idx = out.sort(0, True)
    correct += (predicted == labels).sum().item()
    # print(out.shape, labels.shape)
    correct5 += np.equal(np.argsort(h_x)[:, -5:], labels[:, None]).any(axis=1).sum().item()

print("Top 1 acc:", correct/total)
print("Top 5 acc:", correct5/total)