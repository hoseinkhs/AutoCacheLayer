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
import copy
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import time
import pandas as pd
import numpy as np
from modelwrappers import FaceModel, PlaceModel
from aux import train_one_epoch, get_lr, get_n_params
from meters import ModelMeter, ExitMeter
from server import Server
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from deepspeed.profiling.flops_profiler import get_model_profile

from pytorch_memlab import MemReporter

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

from model_evaluator import evaluate_model

import nni

from torchvision import transforms
from torch.utils.data import DataLoader

def experiment(conf, model, train_data, test_data, return_artifacts=False):
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
    print("Original #Params:", get_n_params(backbone))
    if conf.train_epochs or conf.run_server:
        cache_exits = [
            ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file,
                            experiment=conf.experiment, backbone=conf.backbone_type).get_classifier()
            for i in cached_layers]
    else:
        cache_exits = [
            ClassifierModelLoader(
                ClassifierFactory(f'{conf.exit_type}_exit_{i}', conf.exit_conf_file, experiment=conf.experiment, backbone=conf.backbone_type)).load_model_default(
                    os.path.join(conf.exit_model_path, f"Exit_{i}.pt")
            ) for i in cached_layers]

    nc_model = copy.deepcopy(model)
    nc_model = nc_model.to(conf.test_device)

    model.backbone.set_exit_models(cache_exits)
    model = model.to(conf.test_device)
    print("Cached #Params:", get_n_params(model))
    return 
    if conf.pre_evaluate_backbone:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for images, labels in test_loader:
                # if not total:
                #     print(labels)
                images = images.to(conf.test_device)
                out, _ = model.forward(images, conf, cache=False)
                confidence, predicted = torch.max(out, 1)
                predicted = predicted.to('cpu')
                correct += (predicted == labels).sum().item()
                total+=images.size(0)
        confirm = input(
            f"Original backbone stats: {correct} correct out of {total}, proceed? [Y/n]")
        if confirm == "Y":
            print("Backbone evaluation confirmed successfully, proceeding...")
        else:
            print("EXITING NOW!")
            exit()
            
    
    
    if conf.train_epochs > 0 and not conf.run_server:
        model = model.to(conf.train_device)
        print("Model ready to train:")
        criterion = torch.nn.KLDivLoss(log_target=True).cuda(conf.train_device)
        for num_exit in range(num_exits): #[3, 2, 1, 0]:#
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
    print(f"Testing on {conf.test_device}, Shrink enabled: {conf.shrink}")
    model = model.to(conf.test_device)
    model.eval()
    model.backbone.eval()
    for ex in cache_exits:
        ex.eval()

    if return_artifacts:
        return model, test_data
    
    exits_df = pd.DataFrame()
    model_df = pd.DataFrame()
    test_confidences = [1]#[i/100 for i in range(0, 101, 2)]#
    batch_sizes = [128] #[1, 4, 8, 16, 32, 64, 128] #[1, 4, 8] #[args.test_batch_size] #[128]#[1]#

    if conf.test_num_threads:
        torch.set_num_threads(conf.test_num_threads)
    if conf.test_device == 'cuda:0':
        model = torch.nn.DataParallel(model)
    with torch.no_grad():
        nc_mem_profile = MemReporter(nc_model)
        mem_profile = MemReporter(model)
        for confidence in test_confidences:
            print(f"*********** Confidence: {confidence} ********")
            for batch_size in batch_sizes:
                print(f"------------ Batch Size: {batch_size} ----------")
                for rep in range(conf.repetition):
                    if conf.run_profiler:
                        test_loader = DataLoader(test_data,
                                    batch_size, False, num_workers=0, pin_memory=True)
                        print("RUNNING NC_PROFILER")                        
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as nc_prof:
                            batch_idx = 0
                            for images, labels in test_loader:
                                images = images.to(conf.test_device)        
                                nc_out, nc_results = nc_model.forward(images, conf, cache=False, return_cc=True)
                                print(f"batch_size: {batch_size}, batch_idx: {batch_idx}", end="\r")
                                batch_idx+=1
                                if batch_size * batch_idx > 1:
                                    break
                                # break
                        print("RUNNING CACHED_PROFILER")
                        test_loader = DataLoader(test_data,
                                    batch_size, False, num_workers=0, pin_memory=True)
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                            batch_idx = 0
                            for images, labels in test_loader:
                                images = images.to(conf.test_device)        
                                out, results = model.forward(images, conf, cache=True, threshold=confidence, return_cc=True)
                                print(f"batch_size: {batch_size}, batch_idx: {batch_idx}", end="\r")
                                batch_idx+=1
                                if batch_size * batch_idx > 1:
                                    break
                        print(nc_prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=1))
                        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=1))

                    if conf.run_meters: 
                        batch_idx = 0
                        test_loader = DataLoader(test_data,
                                    batch_size, True, num_workers=0)
                        mm = ModelMeter("Cached")
                        nc_mm = ModelMeter("Original")
                        ems = [ExitMeter(cached_layers[i] if i < num_exits else -1, i, mm) for i in range(num_exits + 1)]
                        for images, labels in test_loader:
                            images = images.to(conf.test_device)
                            nc_model.set_defaults(conf, False, 1, True)
                            nc_out, nc_results = nc_model.forward(images, conf, cache=False, return_cc=True)
                            if conf.count_flops:
                                nc_flops, _, _ = get_model_profile(model=nc_model.backbone, # model
                                                    # input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                    args=[images], # list of positional arguments to the model.
                                                    # kwargs=None, # dictionary of keyword arguments to the model.
                                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                                    detailed=False, # print the detailed profile
                                                    module_depth=1, # depth into the nested modules, with -1 being the inner most modules
                                                    top_modules=1, # the number of top modules to print aggregated profile
                                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                                    as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                                    ignore_modules=None)
                            nc_mm.batch_update(nc_out, labels, nc_results, nc_flops if conf.count_flops else 0)

                            model.set_defaults(conf, True, confidence, True)
                            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                                with record_function("model_inference"):
                                    out, results = model.forward(images, conf, cache=True, threshold=confidence, return_cc=True)
                            if conf.count_flops:
                                flops, _, _ = get_model_profile(model=model.backbone, # model
                                                    # input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                                    args=[images], # list of positional arguments to the model.
                                                    # kwargs=None, # dictionary of keyword arguments to the model.
                                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                                    detailed=True, # print the detailed profile
                                                    module_depth=1, # depth into the nested modules, with -1 being the inner most modules
                                                    top_modules=1, # the number of top modules to print aggregated profile
                                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                                    as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                                    ignore_modules=None)
                            
                            for i in range(num_exits + 1): 
                                # print(len(results["idxs"]), i) 
                                if i < len(results["idxs"]):
                                    ems[i].batch_update(labels, results, nc_out)
                            
                            mm.batch_update(out, labels, results, flops if conf.count_flops else 0)
                            if batch_idx == 0:
                                break
                            batch_idx +=1
                
                        print(mm)    
                        print(nc_mm)
                        model_record = {
                            "Rep": rep,
                            "Confidence": confidence,
                            "BatchSize": batch_size,
                            "ResponseTime": round(nc_mm.time, 4),
                            "CachedResponseTime": round(mm.time, 4),
                            "MTTR": round(nc_mm.time/nc_mm.num_batch, 4),
                            "CachedMTTR": round(mm.samplewise_hit_time/mm.total, 4),
                            "Accuracy": round(100 * nc_mm.correct / nc_mm.total, 2),
                            "CacheAccuracy": -1 if sum([m.hit_count for m in ems][:-1]) == 0 else round(100 * sum([m.cached_correct for m in ems][:-1]) / sum([m.hit_count for m in ems][:-1]), 2),
                            "CachedAccuracy": round(100 * sum([m.correct for m in ems]) / mm.total, 2),
                            "MTTRRatio": round(100 * (mm.samplewise_hit_time)/mm.total/(nc_mm.time/mm.num_batch), 2),
                            "SamplesReachedEnd": ems[-1].num_sample,
                            "BatchesReachedEnd": ems[-1].num_batch,
                            "CudaTime": nc_mm.cuda_time,
                            "CachedCudaTime": mm.cuda_time,
                            # "Mem_A": mem_A,
                            # "Mem_B": f"{mem_B - mem_A} - {mem_B}",
                            # "Mem_C": f"{mem_C - mem_A} - {mem_C}"
                        }
                        if conf.count_flops:
                            model_record.update({
                                "Flops": nc_mm.flops,
                                "CachedFlops": mm.flops,
                                "AvgFlops": nc_mm.flops/nc_mm.total,
                                "AvgCachedFlops": mm.flops/mm.total,
                            })
                        model_df = model_df.append(model_record, ignore_index=True)

                        # print(
                        #     f'Models samplewise MTTR: Cached {sum(samplewise_hit_times)/total:.4f}, Non-cached: {nc_total_time/num_batch:.4f}, ratio:{100 * (sum(samplewise_hit_times)/total)/(nc_total_time/num_batch):.2f} %')
                        for i in range(num_exits+1):
                            em = ems[i]
                            print(em)
                            row = em.__dict__()
                            row.update({
                                "Confidence": confidence,
                                "ExitNumber": i,
                                "BatchSize": batch_size,
                                "Rep": rep
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
    """Preparing place model and data for caching procedure.  
    """
    print(f"Loading {conf.backbone_type} model for experiment")
    train_data = PlaceDataset(
        conf.data_root, conf.train_file, conf.classes_file, limit_per_class=10)
    
    test_data = PlaceDataset(
        conf.data_root, conf.test_file, conf.classes_file, skip_per_class=10)


    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file)
    backbone_model = backbone_factory.get_backbone()

    model = PlaceModel(backbone_model)
    return experiment(conf, model,
               train_data, test_data, return_artifacts)


def face_experiment(conf, return_artifacts=False):
    """Preparing face model and data for caching procedure.
    """
    train_data = ImageDataset(conf.data_root, conf.train_file, classes_file=conf.classes_file, name_as_label=True)
    test_data = ImageDataset(conf.data_root, conf.test_file, classes_file=conf.classes_file, name_as_label=True, allow_unknown=True)

    classifier_factory = ClassifierFactory(
        conf.classifier_type, conf.classifier_conf_file, experiment="Face", backbone=conf.backbone_type)
    classifier_loader = ClassifierModelLoader(classifier_factory)
    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file, experiment="Face")

    model_loader = ModelLoader(backbone_factory)
    backbone_model = model_loader.load_model(conf.backbone_model_path)
    classifier_model = classifier_loader.load_model(conf.classifier_model_path)

    model = FaceModel(backbone_model, classifier_model)
    # print(backbone_model)
    # exit()
    return experiment(conf, model,
               train_data, test_data, return_artifacts)



def cifar100_experiment(conf, return_artifacts=False):
    """Preparing cifar100 model and data for caching procedure.
    """
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data = torchvision.datasets.CIFAR100(root=conf.data_root, train= False, transform= transform_test, download = True)

    train_data, test_data = torch.utils.data.random_split(data, [7000, 3000], generator=torch.Generator().manual_seed(42))
    
    
    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file)

    model_loader = ModelLoader(backbone_factory)

    weights_file = f'./models/cifar100/{conf.backbone_type.lower()}.pt'

    model = backbone_factory.get_backbone()
    model.load_state_dict(torch.load(weights_file))
    test_loader = DataLoader(test_data,
                                   conf.test_batch_size, True, num_workers=0)
    print(next(iter(test_loader))[0].shape)
    # images, labels = next(iter(test_loader))
    # tm = torchvision.models.resnet50(num_classes=100)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                             profile_memory=True) as prof:
    #     model(images)
    # flops, macs, params = get_model_profile(model=model, # model
    #                                         # input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
    #                                         args=[images], # list of positional arguments to the model.
    #                                         # kwargs=None, # dictionary of keyword arguments to the model.
    #                                         print_profile=True, # prints the model graph with the measured profile attached to each module
    #                                         detailed=False, # print the detailed profile
    #                                         module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
    #                                         top_modules=1, # the number of top modules to print aggregated profile
    #                                         warm_up=10, # the number of warm-ups before measuring the time of each module
    #                                         as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
    #                                         output_file=None, # path to the output file. If None, the profiler prints to stdout.
    #                                         ignore_modules=None)
    # print("!!!!!!!!!!", flops, macs, params)
    # print("^^^^^^^^^^", prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=-1))
    # return
    return experiment(conf, PlaceModel(model), 
               test_data, test_data, return_artifacts)


def cifar10_experiment(conf, return_artifacts=False):
    """Preparing cifar10 model and data for caching procedure.
    """
    data = torchvision.datasets.CIFAR10(root=conf.data_root, train= False, transform= torchvision.transforms.ToTensor(), download = True)
    train_data, test_data = torch.utils.data.random_split(data, [5000, 5000], generator=torch.Generator().manual_seed(42))

    backbone_factory = BackboneFactory(
        conf.backbone_type, conf.backbone_conf_file, experiment="Cifar10")

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
    return experiment(conf, PlaceModel(model), 
            test_data, test_data, return_artifacts)




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
    conf.add_argument("--test_num_threads", type=int, default=0,
                      help="The number of threads to use while testing the models.")
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
    conf.add_argument('--repetition', type=int, default=1,
                      help='The evaluation repetition count')
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
    conf.add_argument('--pre_evaluate_backbone', action='store_true', default=False,
                      help='Evaluate backbone\'s accuracy before proceeding')
    conf.add_argument('--run_profiler', '-p', action='store_true', default=False,
                      help='Whether to run flops analysis.')
    conf.add_argument('--run_meters', '-m', action='store_true', default=False,
                      help='Whether to run flops analysis.')
    conf.add_argument('--count_flops', '-f', action='store_true', default=False,
                      help='Whether to run flops analysis.')
    conf.add_argument('--exit_on_all_resolved', action='store_true', default=False,
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
        "Cifar10": cifar10_experiment,
        "Cifar100": cifar100_experiment
    }

    if args.run_server:
        server = Server(args, experiments[args.experiment](args, return_artifacts=True))
    else:
        experiments[args.experiment](args)
    logger.info('Optimization done!')
