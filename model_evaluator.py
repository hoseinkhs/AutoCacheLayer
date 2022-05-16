

def get_model_evaluator(conf, backbone, num_exit, exit_layer, get_loaders, device):
    def train_epoch(cache_model, train_loader, optimizer, epoch, backbone):
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
    def test_epoch(cache_model, test_loader, backbone):
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

    def evaluate_model(model_cls):
        # "model_cls" is a class, need to instantiate
        train_loader, test_loader = get_loaders()
        exit_model = model_cls()
        exit_model.to(device)

        backbone.eval()
        backbone.to(device)

        for p in backbone.parameters():
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
                            exit_model, num_exit, idx, logger)
            lr_schedule.step()
            accuracy = test_epoch(model, exit_model, test_loader)
            nni.report_intermediate_result(accuracy)

        # report final test result
        nni.report_final_result(accuracy)
    return evaluate_model
