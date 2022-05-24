import torch
from torch import optim
from utils.AverageMeter import AverageMeter
import nni
def train_epoch(conf, backbone, cache_model, train_loader, optimizer, epoch, device):
    loss_fn = torch.nn.KLDivLoss(log_target=True).cuda(device)
    cache_model.train()
    for batch_idx, (samples, label) in enumerate(train_loader):
        samples, label = samples.to(device), label.to(device)
        optimizer.zero_grad()
        target, results = backbone(samples, conf, training=True, return_vectors = True)
        vectors = results.report()['vectors'][0]
        cache_output = cache_model(vectors)
        loss = loss_fn(cache_output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(samples), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def test_epoch(conf, backbone, cache_model, test_loader, device):
    cache_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for samples, target in test_loader:
            samples, label = samples.to(device), target.to(device)
            target, results = backbone(samples, conf, training=True, return_vectors = True)
            vectors = results.report()['vectors'][0]
            cache_output = cache_model(vectors)
            # output = model(data)
            pred = cache_output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target).sum().item()

def evaluate_model(conf, get_backbone, num_exit, exit_layer, get_loaders, device, model_cls):
    # "model_cls" is a class, need to instantiate
    backbone = get_backbone()
    backbone.eval()
    backbone.to(device)
    
    cache_model = model_cls()
    cache_model.to(device)

    train_loader, test_loader = get_loaders()

    for p in backbone.parameters():
            p.requires_grad = False
    for p in cache_model.parameters():
        p.requires_grad = True

    parameters = [p for p in cache_model.parameters()
                    if p.requires_grad]

    optimizer = optim.SGD(parameters, lr=conf.lr,
                            momentum=conf.momentum, weight_decay=1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=conf.milestones, gamma=0.1)
    loss_meter = AverageMeter()
    cache_model.train()
    for epoch in range(conf.train_epochs):
        train_epoch(conf, backbone, cache_model, train_loader, optimizer, epoch, device)
        lr_schedule.step()
        accuracy = test_epoch(conf, backbone, cache_model, test_loader, device)
        nni.report_intermediate_result(accuracy)
        # nni.report_intermediate_result(0)

    # report final test result
    nni.report_final_result(accuracy)
    # nni.report_final_result(0.5)

