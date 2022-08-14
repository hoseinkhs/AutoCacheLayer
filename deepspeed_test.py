import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from deepspeed.profiling.flops_profiler import get_model_profile

# from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter
import torchvision

model = models.resnet18(num_classes=100)

device = "cuda:0"
data = torchvision.datasets.CIFAR100(root="./data/cifar10", train= False, transform= torchvision.transforms.ToTensor(), download = True)
train_data, test_data = torch.utils.data.random_split(data, [5000, 5000], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_data, 128, True, num_workers=0)
model = model.to(device)
model.eval()
with torch.no_grad():
    with torch.cuda.device(0):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], #if device == "cpu" else 
                profile_memory=True) as prof:
    # reporter = MemReporter(model)
            idx = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                get_model_profile(model=model, # model
                # input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                args=[inputs], # list of positional arguments to the model.
                # kwargs=None, # dictionary of keyword arguments to the model.
                print_profile=True, # prints the model graph with the measured profile attached to each module
                detailed=True, # print the detailed profile
                module_depth=1, # depth into the nested modules, with -1 being the inner most modules
                top_modules=1, # the number of top modules to print aggregated profile
                warm_up=10, # the number of warm-ups before measuring the time of each module
                as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                output_file=None, # path to the output file. If None, the profiler prints to stdout.
                ignore_modules=None)
                if idx == 0:
                    break
                idx+=1
                
    # reporter.report()

# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
# get_model_profile(model=model.backbone, # model
#                 # input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
#                 args=[images], # list of positional arguments to the model.
#                 # kwargs=None, # dictionary of keyword arguments to the model.
#                 print_profile=True, # prints the model graph with the measured profile attached to each module
#                 detailed=True, # print the detailed profile
#                 module_depth=1, # depth into the nested modules, with -1 being the inner most modules
#                 top_modules=1, # the number of top modules to print aggregated profile
#                 warm_up=10, # the number of warm-ups before measuring the time of each module
#                 as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
#                 output_file=None, # path to the output file. If None, the profiler prints to stdout.
#                 ignore_modules=None)