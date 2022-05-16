"""
@author: Amin Abedi
@date: 20211019
@contact: mohammadamin.abedi@ucalgary.ca
"""
import sys
# sys.path.append('../../')
from test_protocol.utils.model_loader import ClassifierModelLoader, ModelLoader
from classifier.classifier_def import ClassifierFactory
from backbone.backbone_def import BackboneFactory
from data_processor.train_dataset import ImageDataset, PlaceDataset, transform as aug_tf
from utils.AverageMeter import AverageMeter
from concurrent.futures import thread
import os
from datetime import datetime
import torchvision
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
import numpy as np
from modelwrappers import FaceModel, PlaceModel
from aux import train_one_epoch, get_lr
from meters import ModelMeter, ExitMeter
from server import Server

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

from model_evaluator import get_model_evaluator

import nni

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import nni.retiarii.strategy as strategy
search_strategy = strategy.Random(dedup=True)
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig








def experiment(conf, model, criterion, ft_criterion, train_data, test_data, return_artifacts=False):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H:%M:%S")

    conf.train_device = torch.device(conf.train_device)
    conf.test_device = torch.device(conf.test_device)

    train_loader = DataLoader(train_data,
                                   conf.batch_size, True, num_workers=0)
    test_loader = DataLoader(test_data,
                                   conf.test_batch_size, True, num_workers=0)
    cached_layers = model.backbone.cached_layers
    num_exits = len(cached_layers)
    backbone = model.backbone
    if conf.search_cache_models:
        print("SEARCHING FOR BEST CACHE MODEL")
        from modelspace import get_model_space
        spc = get_model_space()
        print(spc.__module__)
        for num_exit in range(num_exits): #[3, 0, 1, 2]:#
            exit_layer = cached_layers[num_exit]
            evaluator = get_model_evaluator(conf, model, num_exit, exit_layer, train_loader, test_loader)
            evaluator = FunctionalEvaluator(evaluator)
            exp = RetiariiExperiment(spc, evaluator, [], search_strategy)
            exp_config = RetiariiExeConfig('local')
            exp_config.experiment_name = f'{conf.backbone_type}-exit{num_exit}-layer{exit_layer}'
            exp_config.max_trial_number = 5   # spawn 4 trials at most
            exp_config.trial_concurrency = 2  # will run two trials concurrently
            exp_config.trial_gpu_number = 1
            exp_config.training_service.use_active_gpu = True
            exp.run(exp_config, 8081)
    if conf.train_epochs or conf.run_server:

        cache_exits = [
            ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file,
                            for_backbone=conf.backbone_type).get_classifier()
            for i in cached_layers]
    else:
        cache_exits = [
            ClassifierModelLoader(
                ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file, for_backbone=conf.backbone_type)).load_model_default(
                    os.path.join(conf.exit_model_path, f"Exit_{i}.pt")
            ) for i in cached_layers]

    
    backbone.set_exit_models(cache_exits)

    
    

    

    # for ii in model.backbone.children():
    #     # for ii in range(len(child)):
    #     if 'BatchNorm' in type(ii).__name__:
    #         ii.track_running_stats = False
    #         print("FOUND ONE", ii)

    model = model.to(conf.test_device)
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            if not total:
                print(labels)
            images = images.to(conf.test_device)
            out, _ = model.forward(images, conf, cache=False)
            confidence, predicted = torch.max(out, 1)
            predicted = predicted.to('cpu')
            correct += (predicted == labels).sum().item()
            total+=images.size(0)
            
    print("INITIAL STATS:", correct, len(test_data), total)
    return
    
    if conf.train_epochs > 0 and not conf.run_server:
        model = model.to(conf.train_device)
        print("Model ready to train:")
        
        for num_exit in range(num_exits): #[3, 0, 1, 2]:#
            exit_model = cache_exits[num_exit]
            exit_layer = cached_layers[num_exit]
            for p in model.parameters():
                p.requires_grad = False
            for p in exit_model.parameters():
                p.requires_grad = True

            parameters = [p for p in exit_model.parameters()
                          if p.requires_grad]

            optimizer = optim.SGD(parameters, lr=conf.lr,
                                  momentum=conf.momentum, weight_decay=1e-4)
            lr_schedule = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=conf.milestones, gamma=0.1)
            loss_meter = AverageMeter()
            exit_model.train()
            for epoch in range(conf.train_epochs):
                train_one_epoch(train_loader, model, optimizer,
                                criterion, epoch, loss_meter, conf,
                                exit_model, num_exit, exit_layer, logger)
                lr_schedule.step()

            if conf.fine_tune > 0:
                for p in model.parameters():
                    p.requires_grad = True
                parameters = [p for p in model.parameters() if p.requires_grad]
                print("Fine tuning, num of params:", len(parameters))
                optimizer = optim.SGD(parameters, lr=conf.lr * 1e-3,
                                      momentum=conf.momentum, weight_decay=1e-4)
                lr_schedule = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=conf.milestones, gamma=0.1)
                loss_meter = AverageMeter()
                for epoch in range(conf.fine_tune):
                    train_one_epoch(train_loader, model, optimizer,
                                    ft_criterion, epoch, loss_meter, conf,
                                    exit_model, num_exit, idx, logger, fine_tuning=True)
                    lr_schedule.step()
            for p in exit_model.parameters():
                p.requires_grad = False

    print(f"Testing on {conf.test_device}, Shrink enabled: {conf.shrink}")
    model = model.to(conf.test_device)
    model.eval()
    model.backbone.eval()
    for ex in cache_exits:
        ex.eval()

    if return_artifacts:
        return model, test_data
    
    exits_df = pd.DataFrame(columns=["Confidence", "ExitNumber", "ExitName", "HitTime",
                            "HitRateOverAll", "HitRate", "Accuracy", "CacheAccuracy", "SamplesReached"])
    model_df = pd.DataFrame(columns=["Confidence", "ResponseTime", "CachedResponseTime",
                            "MTTR", "CachedMTTR", "Accuracy", "CachedAccuracy", "MTTRRatio"])
    batch_df = pd.DataFrame(columns=["BatchSize", "Confidence", "ResponseTime",
                            "CachedResponseTime", "MTTR", "CachedMTTR", "MTTRRatio"])
    test_confidences = [i/100 for i in range(0, 101, 2)]#[.45]#
    test_batch_on_confidences = []  # [0.05, .45]
    
    with torch.no_grad():
        for confidence in test_confidences:
            mm = ModelMeter("Cached")
            nc_mm = ModelMeter("Original")
            ems = [ExitMeter(cached_layers[i] if i < num_exits else -1, i, mm) for i in range(num_exits + 1)]

            for images, labels in test_loader:
                images = images.to(conf.test_device)

                nc_out, nc_results = model.forward(images, conf, cache=False)
                nc_mm.batch_update(nc_out, labels, nc_results)

                out, results = model.forward(images, conf, cache=True, threshold=confidence)
                
                for i in range(num_exits + 1): 
                    # print(len(results["idxs"]), i) 
                    if i < len(results["idxs"]):
                        ems[i].batch_update(labels, results, nc_out)
                    
                mm.batch_update(out, labels, results)

            print(f"*********** Confidence: {confidence} ********")
            print(mm)    
            print(nc_mm)

            model_df = model_df.append({
                "Confidence": confidence,
                "ResponseTime": round(nc_mm.time, 4),
                "CachedResponseTime": round(mm.time, 4),
                "MTTR": round(nc_mm.time/nc_mm.num_batch, 4),
                "CachedMTTR": round(mm.samplewise_hit_time/mm.total, 4),
                "Accuracy": round(100 * nc_mm.correct / nc_mm.total, 2),
                "CacheAccuracy": -1 if sum([m.hit_count for m in ems][:-1]) == 0 else round(100 * sum([m.cached_correct for m in ems][:-1]) / sum([m.hit_count for m in ems][:-1]), 2),
                "CachedAccuracy": round(100 * sum([m.correct for m in ems]) / mm.total, 2),
                "MTTRRatio": round(100 * (mm.samplewise_hit_time)/mm.total/(nc_mm.time/mm.num_batch), 2),
                "SamplesReachedEnd": ems[-1].num_sample,
                "BatchesReachedEnd": ems[-1].num_batch
            }, ignore_index=True)

            # print(
            #     f'Models samplewise MTTR: Cached {sum(samplewise_hit_times)/total:.4f}, Non-cached: {nc_total_time/num_batch:.4f}, ratio:{100 * (sum(samplewise_hit_times)/total)/(nc_total_time/num_batch):.2f} %')
            for i in range(num_exits+1):
                em = ems[i]
                print(em)
                row = em.__dict__()
                row.update({
                    "Confidence": confidence,
                    "ExitNumber": i
                })
                exits_df = exits_df.append(row, ignore_index=True)

            # for i in range(num_exits+1):
            #     try:
            #         print(
            #             f'EXIT {i} | Acc: {100 * correct[i] / hit_counts[i]:.2f}%, Cache Acc: {100 * cached_correct[i] / hit_counts[i]:.2f}%, HR: {100 * hit_counts[i] / total:.2f}, Confidence: {cache_confidences[i]/num_sample_exit[i]:.3f}, out of: {total} (batch size: {conf.batch_size})')
            #     except ZeroDivisionError:
            #         pass
            exits_df.astype({c: 'int32' for c in ["ExitNumber", "ExitName", "SamplesReached"]}, copy=False).to_csv(
                f"{conf.report_dir}/exits.csv", index_label="Idx")
            model_df.to_csv(
                f"{conf.report_dir}/model.csv", index_label="Idx")


def place_experiment(conf, return_artifacts=False):
    """Preparing place model, criterion and data for caching procedure.  
    """
    print(f"Loading {conf.backbone_type} model for experiment")
    train_data = PlaceDataset(
        conf.data_root, conf.train_file, conf.classes_file, limit_per_class=10)
    
    test_data = PlaceDataset(
        conf.data_root, conf.test_file, conf.classes_file, skip_per_class=10)

    criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
    ft_criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)

    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file)
    backbone_model = backbone_factory.get_backbone()

    model = PlaceModel(backbone_model)
    return experiment(conf, model, criterion, ft_criterion,
               train_data, test_data, return_artifacts)


def face_experiment(conf, return_artifacts=False):
    """Preparing face model, criterion and data for caching procedure.
    """
    train_data = ImageDataset(conf.data_root, conf.train_file, classes_file=conf.classes_file, name_as_label=True)
    test_data = ImageDataset(conf.data_root, conf.test_file, classes_file=conf.classes_file, name_as_label=True, allow_unknown=True)
    
    criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
    ft_criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)

    classifier_factory = ClassifierFactory(
        conf.classifier_type, conf.classifier_conf_file)
    classifier_loader = ClassifierModelLoader(classifier_factory)
    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file)

    model_loader = ModelLoader(backbone_factory)
    backbone_model = model_loader.load_model(conf.backbone_model_path)
    classifier_model = classifier_loader.load_model(conf.classifier_model_path)

    model = FaceModel(backbone_model, classifier_model)
    # print(backbone_model)
    # exit()
    return experiment(conf, model, criterion,
               ft_criterion, train_data, test_data, return_artifacts)

def cifar10_experiment(conf, return_artifacts=False):
    """Preparing face model, criterion and data for caching procedure.
    """
    test_data = torchvision.datasets.CIFAR10(root=conf.data_root, train= False, transform= torchvision.transforms.ToTensor(), download = True)
    # print(len(data))
    # data = torchvision.datasets.CIFAR10(root=conf.data_root, train= False, transform= T, download = True)
    # train_data, test_data = torch.utils.data.random_split(data, [3000, 7000], generator=torch.Generator().manual_seed(42))
    criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
    ft_criterion = torch.nn.CrossEntropyLoss().cuda(conf.train_device)

    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file)

    model_loader = ModelLoader(backbone_factory)

    weights_file = f'./models/cifar10/{conf.backbone_type.lower()}.pt'

    model = backbone_factory.get_backbone()
    model.load_state_dict(torch.load(weights_file))
    # model.train()
    # model.to(conf.train_device)
    # for p in model.parameters():
    #                 p.requires_grad = True
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # print("Fine tuning, num of params:", len(parameters))
    # optimizer = optim.SGD(parameters, lr=conf.lr * 1e-3,
    #                         momentum=conf.momentum, weight_decay=1e-4)
    # lr_schedule = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=conf.milestones, gamma=0.1)
    # loss_meter = AverageMeter()
    # train_loader = DataLoader(train_data,
    #                                conf.batch_size, True, num_workers=0)
    # for cur_epoch in range(10):   
    #     for batch_idx, (images, labels) in enumerate(train_loader):
    #         images = images.to(conf.train_device)
    #         labels = labels.to(conf.train_device)
    #         # print(images.size(), labels.size())
    #         labels = labels.squeeze()

    #         outputs, _ = model.forward(images, conf, training=True)
    #         loss = ft_criterion(outputs, labels) 
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         loss_meter.update(loss.item(), images.shape[0])
    #         if batch_idx % conf.print_freq == 0:
    #             loss_avg = loss_meter.avg
    #             lr = get_lr(optimizer)
    #             logger.info('FT Epoch %d, iter %d/%d, lr %f, loss %f' % 
    #                         (cur_epoch, batch_idx, len(train_loader), lr, loss_avg))
    #             loss_meter.reset()
    #     lr_schedule.step()
    # torch.save(model.state_dict(), weights_file)
    return experiment(conf, PlaceModel(model), criterion,
               ft_criterion, test_data, test_data, return_artifacts)




if __name__ == '__main__':
    conf = argparse.ArgumentParser(
        description='cache_training for face recognition models.')
    conf.add_argument("--experiment", type=str,
                      help="Face recognition or Place classification (Face|Place)")
    conf.add_argument("--data_root", type=str,
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      help="The training file path.")
    conf.add_argument("--test_file", type=str,
                      help="The testing file path.")
    conf.add_argument("--train_device", type=str, required=True,
                      help="The device to train the models on.")
    conf.add_argument("--test_device", type=str, required=True,
                      help="The device to test the models on.")
    conf.add_argument("--backbone_type", type=str,
                      help="Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type=str,
                      help="the path of backbone_conf.yaml.")
    conf.add_argument("--backbone_model_path", type=str,
                      help="the path of trained backbone model pt file")
    conf.add_argument("--classifier_type", type=str,
                      help="Dense2Layer only!")
    conf.add_argument("--classifier_conf_file", type=str,
                      help="the path of classifier_conf.yaml.")
    conf.add_argument("--classifier_model_path", type=str,
                      help="the path of trained final classifier model pt file")
    conf.add_argument("--exit_type", type=str,
                      help="type of the exit classifier model")
    conf.add_argument("--exit_conf_file", type=str,
                      help="the path of exit_conf.yaml.")
    conf.add_argument('--exit_model_path', type=str,
                      help='paths to the exit models')
    conf.add_argument('--lr', type=float, default=0.1,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str,
                      help="The folder to save models.")
    conf.add_argument("--trial", type=str, default="000",
                      help="Trial number for reports.")
    conf.add_argument('--train_epochs', type=int, default=9,
                      help='The training epochs.')
    conf.add_argument('--online_train_epochs', type=int, default=9,
                      help='The online training epochs.')
    conf.add_argument('--fine_tune', type=int, default=9,
                      help='Fine tune the whole model after caches are trained')
    conf.add_argument('--step', type=str, default='2,5,7',
                      help='Step for lr.')
    conf.add_argument('--print_freq', type=int, default=10,
                      help='The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=10,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=128,
                      help='The training batch size over all gpus.')
    conf.add_argument('--test_batch_size', type=int, default=1,
                      help='The testing batch size.')
    conf.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for sgd.')
    conf.add_argument('--log_dir', type=str, default='log',
                      help='The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type=str,
                      help='The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type=str, default='mv_epoch_8.pt',
                      help='The path of pretrained model')
    conf.add_argument('--shrink', '-s', action='store_true', default=False,
                      help='Whether shrink the batches upon cache hit.')
    conf.add_argument('--distillation_test', '-d', action='store_true', default=False,
                      help='Testing distillation or the new method')
    conf.add_argument('--classes_file',
                      help='List of names to train the classifier for')
    conf.add_argument('--num_classes', type=int, required=True,
                      help='Number of classes')
    conf.add_argument('--run_server', action='store_true', default=False,
                      help='Run server')
    conf.add_argument('--search_cache_models', action='store_true', default=False,
                      help='Run autoML on cache model architectures')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.exit_model_path):
        os.makedirs(args.exit_model_path)
    args.report_dir = f"{args.out_dir}/reports/{args.exit_type}/{args.test_device}/{args.trial}"
    if not os.path.exists(args.report_dir):
        os.makedirs(args.report_dir)
    elif not args.run_server:
        confirm = input(
            f"The report dir {args.report_dir} already exists, override?[Y/n]")
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

    experiments = {
        "Face": face_experiment,
        "Place": place_experiment,
        "Cifar": cifar10_experiment
    }

    if args.run_server:
        server = Server(args, experiments[args.experiment](args, return_artifacts=True))
    else:
        experiments[args.experiment](args)
    logger.info('Optimization done!')
