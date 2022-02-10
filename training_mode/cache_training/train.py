"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
from concurrent.futures import thread
import os
from datetime import datetime
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time
import pandas as pd
sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset, transform as aug_tf
from backbone.backbone_def import BackboneFactory
from classifier.classifier_def import ClassifierFactory
from test_protocol.utils.model_loader import ClassifierModelLoader, ModelLoader
logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

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

    def forward(self, data, ):
        feat, results = self.backbone.forward(data)
        if feat.shape[0]:
            classification = self.classifier.forward(feat)
            results["outputs"].append(classification)
            results["hits"].append(torch.ones(classification.shape[0]))
            tt = time.time()
            results["hit_times"].append(tt)
            results["end_time"] = tt
            return classification, results
        else:
            results["end_time"] = time.time()
            return feat, results
        

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf, exit_model, exit_idx, fine_tuning=False):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.train_device)
        labels = labels.to(conf.train_device)
        labels = labels.squeeze()

        outputs, results = model.forward(images)
        exits = results["exits"]
        early_pred = exit_model(exits[exit_idx])
        if fine_tuning:
            # _, early_pred = torch.max(early_pred, 1)
            loss = criterion(early_pred, outputs)
        else:
            loss = criterion(early_pred, outputs) 
            # loss = criterion(early_pred, outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Exit %d, Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (exit_idx, cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        # if (batch_idx + 1) % conf.save_freq == 0:
        #     saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
        #     state = {
        #         'state_dict': model.module.state_dict(),
        #         'epoch': cur_epoch,
        #         'batch_id': batch_idx
        #     }
        #     torch.save(state, os.path.join(conf.out_dir, saved_name))
        #     logger.info('Save checkpoint %s to disk.' % saved_name)
    saved_name = 'Exit_%d_epoch_%d.pt' % (exit_idx, cur_epoch)
    state = {'state_dict': exit_model.state_dict(), 
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(f"{conf.out_dir}/exits/{conf.exit_type}", saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def is_confident(x, threshold):
    x_exp = torch.exp(x)
    mx, _ = torch.max(x_exp, dim=1)
    return torch.gt(mx, threshold)

def distillation(y, teacher_scores, labels, T=4, alpha=0):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)


def train(conf):
    """Total training procedure.
    """
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")
    train_data_loader = DataLoader(ImageDataset(conf.data_root, conf.train_file, names_file=conf.names_file, name_as_label=True), 
                             conf.batch_size, True, num_workers = 0)
    test_dataset = ImageDataset(conf.data_root, conf.test_file, names_file=conf.names_file, name_as_label=True)
    test_data_loader = DataLoader(test_dataset, 
                             conf.batch_size, True, num_workers = 0)
    conf.train_device = torch.device(conf.train_device)
    conf.test_device = torch.device(conf.test_device)
    
    # ft_criterion = torch.nn.CrossEntropyLoss().cuda(conf.train_device)
    criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
    ft_criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
    # criterion = distillation
    classifier_factory = ClassifierFactory(conf.classifier_type, conf.classifier_conf_file)
    classifier_loader = ClassifierModelLoader(classifier_factory)
    if conf.train_exits:
        cache_exits = [ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file).get_classifier() for i in range(conf.num_exits)]
    else:
        cache_exits = [ClassifierModelLoader(ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file)).load_model_default(conf.exit_model_paths[i]) for i in range(conf.num_exits)]
    cache_hits = [is_confident for i in range(conf.num_exits)]
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
    model_loader = ModelLoader(backbone_factory, cache_enabled=False, return_exits=True, cache_exits=cache_exits, cache_hits=cache_hits)
    
    backbone_model = model_loader.load_model(conf.backbone_model_path)
    classifier_model = classifier_loader.load_model(conf.classifier_model_path)
    model = FaceModel(backbone_model, classifier_model)
    model = model.to(conf.train_device)
    print("Model ready to train")
    if conf.train_exits:
        for idx in range(3):
            exit_model = cache_exits[idx]
            
            for p in exit_model.parameters():
                p.requires_grad = True
            parameters = [p for p in exit_model.parameters() if p.requires_grad]
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"EX {idx} Params: {num_params}")
            logger.info(exit_model)
            optimizer = optim.SGD(parameters, lr = conf.lr, 
                                momentum = conf.momentum, weight_decay = 1e-4)
            lr_schedule = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones = conf.milestones, gamma = 0.1)
            loss_meter = AverageMeter()
            exit_model.train()
            for epoch in range(conf.epochs):
                train_one_epoch(train_data_loader, model, optimizer, 
                                criterion, epoch, loss_meter, conf,
                                exit_model, idx)
                lr_schedule.step()
    
            if conf.fine_tune > 0:
                for p in model.parameters():
                    p.requires_grad = True
                parameters = [p for p in model.parameters() if p.requires_grad]
                print("Fine tuning, num of params:", len(parameters))
                optimizer = optim.SGD(parameters, lr = conf.lr * 1e-3, 
                                    momentum = conf.momentum, weight_decay = 1e-4)
                lr_schedule = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones = conf.milestones, gamma = 0.1)
                loss_meter = AverageMeter()
                for epoch in range(conf.fine_tune):
                    train_one_epoch(train_data_loader, model, optimizer, 
                        ft_criterion, epoch, loss_meter, conf,
                        exit_model, idx, fine_tuning=True)
                    lr_schedule.step()
                for p in model.parameters():
                    p.requires_grad = False
            for p in exit_model.parameters():
                p.requires_grad = False
    print(f"Testing on {conf.test_device}, Shrink enabled: {conf.shrink}")
    model = model.to(conf.test_device)
    model.eval()
    model.backbone.eval()
    for ex in cache_exits:
        ex.eval()
    exits_df = pd.DataFrame(columns=["Confidence", "ExitNumber", "HitTime", "HitRateOverAll", "HitRate", "Accuracy", "CacheAccuracy", "SamplesReached"])
    model_df = pd.DataFrame(columns=["Confidence", "ResponseTime", "CachedResponseTime", "MTTR", "CachedMTTR", "Accuracy", "CachedAccuracy", "MTTRRatio"])
    batch_df = pd.DataFrame(columns=["BatchSize", "Confidence", "ResponseTime", "CachedResponseTime", "MTTR", "CachedMTTR", "MTTRRatio"])
    test_confidences =  [i/100 for i in range(0, 101, 2)]
    test_batch_on_confidences = []#[0.05, .45]
    
    with torch.no_grad():
        for confidence in test_confidences:
            nc_correct = 0
            correct = [0 for i in range(conf.num_exits + 1)]
            cached_correct = [0 for i in range(conf.num_exits + 1)]
            hit_counts = [0 for i in range(conf.num_exits + 1)]
            hit_times = [0 for i in range(conf.num_exits + 1)]
            samplewise_hit_times = [0 for i in range(conf.num_exits + 1)]
            num_batch_exit = [0 for i in range(conf.num_exits + 1)]
            num_sample_exit = [0 for i in range(conf.num_exits + 1)]
            cache_confidences = [0 for i in range(conf.num_exits + 1)]
            total_time = 0
            nc_total_time = 0
            total = 0
            num_batch = 0
            
            for images, labels in test_data_loader:
                num_batch += 1
                total += labels.size(0)
                images = images.to(conf.test_device)

                model.backbone.config_cache(active=False, threshold = confidence)
                nc_out , nc_results = model(images)
                _, nc_predicted = torch.max(nc_out, 1)
                nc_predicted = nc_predicted.to('cpu')
                nc_correct += (nc_predicted == labels).sum().item()
                nc_time = nc_results["end_time"] - nc_results["start_time"]
                nc_total_time += nc_time
                
                model.backbone.config_cache(active=True, shrink=conf.shrink, threshold = confidence)
                _, results = model(images)
                start_time = results["start_time"]
                end_time = results["end_time"]
                tt = end_time - start_time
                total_time += tt

                for i in range(conf.num_exits + 1):
                    idxs = results["idxs"][i]
                    if idxs.shape[0] == 0:
                        print(f"Batch#{num_batch} received empty at exit#{i}")
                        break
                    hits = results["hits"][i]
                    out = results["outputs"][i]
                    hits = hits.to('cpu').bool()
                    idxs = idxs.to('cpu')
                    cache_confidence, cache_predicted = torch.max(out, 1)
                    cache_predicted = cache_predicted.to('cpu')
                    # Results calculations
                    cache_confidences[i] += torch.exp(cache_confidence).sum().item()
                    current_corrects = (cache_predicted[hits] == labels[idxs][hits]).sum().item()
                    # print("Current corrects:", current_corrects)
                    correct[i] += current_corrects
                    cached_correct[i] += (cache_predicted[hits] == nc_predicted[idxs][hits]).sum().item()
                    num_hits = torch.sum(hits).item()
                    hit_time = (results["hit_times"][i] - start_time)
                    hit_counts[i] += num_hits
                    hit_times[i] += hit_time
                    samplewise_hit_times[i] += num_hits * hit_time
                    num_batch_exit[i] +=1
                    num_sample_exit[i] += cache_predicted.shape[0]


            print(f"************RESULTS FOR CONFIDENCE: {confidence}************")
            print(samplewise_hit_times, hit_counts, sum(hit_counts), total)
            print(f'Non-cached model accuracy: {100 * nc_correct / total:.2f} %, out of: {total}')
            print(f'Cached model accuracy: {100 * sum(correct) / total:.2f} %, out of: {total}')

            model_df = model_df.append({
                "Confidence": confidence,
                "ResponseTime":round(nc_total_time, 4),
                "CachedResponseTime": round(total_time, 4),
                "MTTR": round(nc_total_time/num_batch, 4),
                "CachedMTTR": round(sum(samplewise_hit_times)/total, 4),
                "Accuracy": round(100 * nc_correct / total, 2),
                "CacheAccuracy": -1 if sum(hit_counts[:-1]) == 0 else round(100 * sum(cached_correct[:-1]) / sum(hit_counts[:-1]), 2),
                "CachedAccuracy": round(100 * sum(correct) / total, 2),
                "MTTRRatio": round(100 * (sum(samplewise_hit_times)/total)/(nc_total_time/num_batch), 2)
            }, ignore_index=True)

            print(f'Models samplewise MTTR: Cached {sum(samplewise_hit_times)/total:.4f}, Non-cached: {nc_total_time/num_batch:.4f}, ratio:{100 * (sum(samplewise_hit_times)/total)/(nc_total_time/num_batch):.2f} %')
            for i in range(conf.num_exits+1):
                print(f'EXIT {i} | hit times {100 * hit_times[i] / nc_total_time:.2f} % , out of {total_time:.2f} sec, nc time {nc_total_time:.2f}')
                exits_df = exits_df.append(
                    {
                        "Confidence": confidence,
                        "ExitNumber": i,
                        "HitTime": round(100 * hit_times[i] / nc_total_time, 2),
                        "HitRateOverAll": round(100 * hit_counts[i] / total, 2),
                        "HitRate": -1 if num_sample_exit[i] == 0 else round(100 * hit_counts[i] / num_sample_exit[i], 2),
                        "Accuracy": -1 if hit_counts[i] == 0 else round(100 * correct[i] / hit_counts[i], 2),
                        "CacheAccuracy": -1 if hit_counts[i] == 0 else round(100 * cached_correct[i] / hit_counts[i], 2),
                        "SamplesReached": num_sample_exit[i]
                    }, ignore_index=True)

            for i in range(conf.num_exits+1):
                try:
                    print(f'EXIT {i} | Acc: {100 * correct[i] / hit_counts[i]:.2f}%, Cache Acc: {100 * cached_correct[i] / hit_counts[i]:.2f}%, HR: {100 * hit_counts[i] / total:.2f}, Confidence: {cache_confidences[i]/num_sample_exit[i]:.3f}, out of: {total} (batch size: {conf.batch_size})')
                except ZeroDivisionError:
                    pass
            exits_df.to_csv(f"{conf.report_dir}/exits.csv", index_label="Idx")
            model_df.to_csv(f"{conf.report_dir}/model.csv", index_label="Idx")
            
if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='cache_training for face recognition models.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--test_file", type = str,  
                      help = "The testing file path.")
    conf.add_argument("--train_device", type = str, required=True,  
                      help = "The device to train the models on.")
    conf.add_argument("--test_device", type = str, required=True,
                      help = "The device to test the models on.")
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
    conf.add_argument("--classifier_model_path", type = str, 
                      help = "the path of trained final classifier model pt file")
    conf.add_argument("--num_exits", type = int, 
                      help = "number of exit models")
    conf.add_argument("--exit_type", type = str, 
                      help = "type of the exit classifier model")
    conf.add_argument("--exit_conf_file", type = str, 
                      help = "the path of exit_conf.yaml.")
    conf.add_argument('--exit_model_paths', nargs='*',
                      help='paths to the exit models')
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument("--trial", type = str, default="000",
                      help = "Trial number for reports.")
    conf.add_argument('--epochs', type = int, default = 9, 
                      help = 'The training epochs.')
    conf.add_argument('--fine_tune', type = int, default = 9, 
                      help = 'Fine tune the whole model after caches are trained')
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
    conf.add_argument('--train_exits', '-t', action = 'store_true', default = False, 
                      help = 'Whether train exits or to test from a checkpoint.')
    conf.add_argument('--shrink', '-s', action = 'store_true', default = False, 
                      help = 'Whether shrink the batches upon cache hit.')
    conf.add_argument('--names_file', required=True,
                      help = 'List of names to train the classifier for')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args.report_dir = f"{args.out_dir}/reports/{args.exit_type}/{args.test_device}/{args.trial}"
    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)
    else:
        confirm = input(f"The report dir {args.report_dir} already exists, override?[Y/n]")
        if confirm == "n":
            print("EXITING NOW!")
            exit()
        else:
            print("Confirmed successfully, proceeding...")
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')

# Friday ok
# Monday 2:35 - 5
# Tuesday 2-5 + peyman
# Wednesday 3-5
# Thursday 2-3

