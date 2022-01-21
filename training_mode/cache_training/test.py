# """
# @author: Jun Wang
# @date: 20201019
# @contact: jun21wangustc@gmail.com
# """
# import os
# import sys
# import shutil
# import argparse
# import logging as logger

# import torch
# from torch import optim
# from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

# sys.path.append('../../')
# from utils.AverageMeter import AverageMeter
# from data_processor.train_dataset import ImageDataset, transform as aug_tf
# from backbone.backbone_def import BackboneFactory
# from classifier.classifier_def import ClassifierFactory
# from test_protocol.utils.model_loader import ClassifierModelLoader, ModelLoader
# logger.basicConfig(level=logger.INFO, 
#                    format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
#                    datefmt='%Y-%m-%d %H:%M:%S')

# class EarlyExitFaceModel(torch.nn.Module):
#     """Define a classification face model which contains a backbone and a classifier.
    
#     Attributes:
#         backbone(object): the backbone of face model.
#         classifier(object): the classifier of face model.
#     """
#     def __init__(self, backbone_model, classifier_model):
#         """Init face model by backbone factorcy and classifier factory.
        
#         Args:
#             backbone_factory(object): produce a backbone according to config files.
#             classifier_factory(object): produce a classifier according to config files.
#         """
#         super(EarlyExitFaceModel, self).__init__()
#         self.backbone = backbone_model
#         self.classifier = classifier_model
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = False

#     def forward(self, data, ):
#         out, cache_hit = self.backbone.forward(data)
#         if cache_hit!=-1:
#             return out, 1
#         classification = self.classifier.forward(out)
#         return classification, 0

# def is_confident(x):
#     print("CONFIDENCE: ", x.size())
#     x_exp = torch.exp(x)
#     mx, idx = torch.max(x_exp, dim=1)
    
#     print("mx:", mx)
#     print("exp:", x_exp[0])
#     print("gt:", torch.gt(mx, 0.8))
#     print("idxs:", idx)
#     return torch.gt(mx, 0.8)

# def test(conf):
#     """Total training procedure.
#     """
#     train_data_loader = DataLoader(ImageDataset(conf.data_root, conf.train_file, names_file=conf.names_file, name_as_label=True), 
#                              conf.batch_size, True, num_workers = 0)
#     test_data_loader = DataLoader(ImageDataset(conf.data_root, conf.test_file, names_file=conf.names_file, name_as_label=True), 
#                              conf.batch_size, True, num_workers = 0)
#     conf.device = torch.device('cuda:0')
    
#     classifier_factory = ClassifierFactory(conf.classifier_type, conf.classifier_conf_file)
#     classifier_loader = ClassifierModelLoader(classifier_factory)
#     print(args.exit_models_paths)
#     cache_exits = [
#         ClassifierModelLoader(ClassifierFactory(f'{conf.exit_type}_exit_{i+1}', conf.exit_conf_file)).load_model_default(args.exit_models_paths[0]) if i==2 else ClassifierFactory(f'{conf.exit_type}_exit_{i+1}', conf.exit_conf_file).get_classifier()
#         for i in range(3)]
#     cache_hits = [is_confident for i in range(3)]
    
#     backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
#     model_loader = ModelLoader(backbone_factory, cache_enabled=True, return_exits=False, cache_exits=cache_exits, cache_hits= cache_hits)
    
#     backbone_model = model_loader.load_model(args.backbone_model_path)
#     classifier_model = classifier_loader.load_model(args.classifier_model_path)
#     model = EarlyExitFaceModel(backbone_model, classifier_model)
#     # cache_disabled_model = model
#     # cache_disabled_model.cache_enabled = False
#     model = torch.nn.DataParallel(model).cuda()
#     # cache_disabled_model = 
    
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_data_loader:
#             out, hit = model.forward(images)
#             print(hit)
#             _, predicted = torch.max(out, 1)
#             predicted = predicted.to('cpu')
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # print(predicted[:10], labels[:10], predicted.size())
#             break
#         print('Testing accuracy: {} %'.format(100 * correct / total))                     

# if __name__ == '__main__':
#     conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
#     conf.add_argument("--data_root", type = str, 
#                       help = "The root folder of training set.")
#     conf.add_argument("--train_file", type = str,  
#                       help = "The training file path.")
#     conf.add_argument("--test_file", type = str,  
#                       help = "The testing file path.")
#     conf.add_argument("--backbone_type", type = str, 
#                       help = "Mobilefacenets, Resnet.")
#     conf.add_argument("--backbone_conf_file", type = str, 
#                       help = "the path of backbone_conf.yaml.")
#     conf.add_argument("--backbone_model_path", type = str, 
#                       help = "the path of trained backbone model pt file")
#     conf.add_argument("--classifier_type", type = str, 
#                       help = "Dense2Layer only!")
#     conf.add_argument("--classifier_conf_file", type = str, 
#                       help = "the path of classifier_conf.yaml.")
#     conf.add_argument("--classifier_model_path", type = str, 
#                       help = "the path of trained final classifier model pt file")
#     conf.add_argument("--exit_type", type = str, 
#                       help = "type of the exit classifier model")
#     conf.add_argument("--exit_conf_file", type = str, 
#                       help = "the path of exit_conf.yaml.")
#     conf.add_argument("--exit_models_paths", type = str, nargs='+',
#                       help = "the path of exit_conf.yaml.")
#     conf.add_argument("--out_dir", type = str, 
#                       help = "The folder to save models.")
#     conf.add_argument('--batch_size', type = int, default = 128, 
#                       help='The training batch size over all gpus.')
#     conf.add_argument('--log_dir', type = str, default = 'log', 
#                       help = 'The directory to save log.log')
#     conf.add_argument('--tensorboardx_logdir', type = str, 
#                       help = 'The directory to save tensorboardx logs')
#     conf.add_argument('--names_file', required=True,
#                       help = 'List of names to train the classifier for')
#     args = conf.parse_args()
    
#     if not os.path.exists(args.out_dir):
#         os.makedirs(args.out_dir)
#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#     tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
#     if os.path.exists(tensorboardx_logdir):
#         shutil.rmtree(tensorboardx_logdir)
#     writer = SummaryWriter(log_dir=tensorboardx_logdir)
#     args.writer = writer
    
#     logger.info('Start optimization.')
#     logger.info(args)
#     test(args)
#     logger.info('Optimization done!')
