"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset, transform as aug_tf
from backbone.backbone_def import BackboneFactory
from classifier.classifier_def import ClassifierFactory
from test_protocol.utils.model_loader import ModelLoader
logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class FaceModel(torch.nn.Module):
    """Define a classification face model which contains a backbone and a classifier.
    
    Attributes:
        backbone(object): the backbone of face model.
        classifier(object): the classifier of face model.
    """
    def __init__(self, backbone_model, classifier_factory):
        """Init face model by backbone factorcy and classifier factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            classifier_factory(object): produce a classifier according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = classifier_factory.get_classifier()

    def forward(self, data, ):
        feat, _ = self.backbone.forward(data)
        classification = self.classifier.forward(feat)
        return classification

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()

        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if (batch_idx + 1) % conf.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
    saved_name = 'Epoch_%d.pt' % cur_epoch
    state = {'state_dict': model.module.classifier.state_dict(), 
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def train(conf):
    """Total training procedure.
    """
    train_data_loader = DataLoader(ImageDataset(conf.data_root, conf.train_file, names_file=conf.names_file, name_as_label=True), 
                             conf.batch_size, True, num_workers = 0)
    test_data_loader = DataLoader(ImageDataset(conf.data_root, conf.test_file, names_file=conf.names_file, name_as_label=True), 
                             conf.batch_size, True, num_workers = 0)
    conf.device = torch.device('cuda:0')
    
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
    model_loader = ModelLoader(backbone_factory)
    classifier_factory = ClassifierFactory(conf.classifier_type, conf.classifier_conf_file)
    backbone_model = model_loader.load_model(args.backbone_model_path)
    model = FaceModel(backbone_model, classifier_factory)
    
    ori_epoch = 0
    if conf.resume:
        ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
        state_dict = torch.load(args.pretrain_model)['state_dict']
        model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)
    loss_meter = AverageMeter()
    model.train()
    print(conf.epoches)
    for epoch in range(ori_epoch, conf.epoches):
        train_one_epoch(train_data_loader, model, optimizer, 
                        criterion, epoch, loss_meter, conf)
        lr_schedule.step()   
    with torch.no_grad():
        correct = 0
        total = 0
        val_sum = 0
        for images, labels in test_data_loader:
            # print(images.shape)
            imgs = images
            for i in range(1):
                # print(images.shape)
                # if i>0:
                # for x in range(images.shape[0]):
                #     imgs[x] = aug_tf(images[x])
                out = model(imgs)
                print(out.shape)
                val, predicted = torch.max(out, 1)
                predicted = predicted.to('cpu')
                # val = torch.exp(val)
                val = val.to('cpu')
                print(val)
                total += labels.size(0)
                val_sum += val.sum()
                correct += (predicted == labels).sum().item()
                # print(predicted[:10], labels[:10], predicted.size())

        print('Testing accuracy: {} %, mean confidence: {} %'.format(100 * correct / total, val_sum/total))                     

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--test_file", type = str,  
                      help = "The testing file path.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--backbone_model_path", type = str, 
                      help = "the path of trained backbone model pt file")
    conf.add_argument("--classifier_type", type = str, 
                      help = "Dense2Layer only!")
    conf.add_argument("--classifier_conf_file", type = str, 
                      help = "the path of classifier_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    conf.add_argument('--names_file', required=True,
                      help = 'List of names to train the classifier for')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
