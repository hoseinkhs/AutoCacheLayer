import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

# from pytorch_memlab import LineProfiler
from pytorch_memlab import MemReporter
import torchvision

model = models.resnet18(num_classes=100)

device = "cpu"
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
                model(inputs)
                if idx == 0:
                    break
                idx+=1
                
    # reporter.report()

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))