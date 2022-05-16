from fastapi import FastAPI, WebSocket
import uvicorn
import torch
import time
import json
from torch import optim
from torch.utils.data import DataLoader
import logging
from aux import train_one_epoch, get_lr
from threading import Thread
import copy
from utils.AverageMeter import AverageMeter
import itertools
import numpy as np
import os

class Server():
    def update_model(self, items_count =None):
        self.updating = True
        self.new_items = 0
        item_limit = (items_count or self.seen_items)
        item_limit -= item_limit%1000
        self.logger.info(f"GOING FOR **** {item_limit}")
        self.test_data.item_limit = item_limit
        train_loader = DataLoader(self.test_data, self.conf.batch_size, True, num_workers=0)
        model_copy = copy.deepcopy(self.model).to(self.conf.train_device) #self.model.to(self.conf.train_device) #
        backbone = model_copy.backbone
        criterion = torch.nn.KLDivLoss(log_target=True).to(self.conf.train_device)
        filenames = [f"./server/{self.conf.backbone_type}_exit{num_exit}_{item_limit}.pt" for num_exit in range(len(backbone.cached_layers))]
        for num_exit in range(len(backbone.cached_layers)):
            fn = filenames[num_exit]
            if os.path.exists(fn):
                continue
            exit_model = backbone.cache_exits[num_exit]
            idx = backbone.cached_layers[num_exit]
            for p in model_copy.parameters():
                p.requires_grad = False
            for p in exit_model.parameters():
                p.requires_grad = True

            parameters = [p for p in exit_model.parameters()
                          if p.requires_grad]

            optimizer = optim.SGD(parameters, lr=self.conf.lr,
                                  momentum=self.conf.momentum, weight_decay=1e-4)
            lr_schedule = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.conf.milestones, gamma=0.1)
            loss_meter = AverageMeter()
            exit_model.train()
            for epoch in range(self.conf.online_train_epochs):
                train_one_epoch(train_loader, model_copy, optimizer,
                                criterion, epoch, loss_meter, self.conf,
                                exit_model, num_exit, idx, self.logger, save_model=fn if epoch == self.conf.online_train_epochs -1 else None)
                lr_schedule.step()
        for idx, exit_model in enumerate(self.model.backbone.cache_exits):
            exit_model.load_state_dict(torch.load(filenames[idx])['state_dict'], strict=True) 
        del model_copy
        self.model.version+=1
        self.model_version +=1
        self.logger.info(f"Update done! {self.model_version} {self.model.version}", )
        self.updating = False
    def __init__(self, conf, artifacts):
        torch.cuda.set_device(0)
        self.confidence_threshold = 0.45
        self.logger = logging.getLogger("Model server")
        app = FastAPI()
        model, test_data = artifacts
        self.updating = False
        self.conf = conf
        self.model_version = 0
        self.model = model.to(conf.test_device)
        self.test_data = test_data
        self.sampler = torch.utils.data.RandomSampler(test_data, replacement=True, num_samples=int(1e6))
        self.queue = []
        self.new_items = 0
        self.seen_items = 0
        self.logger.info(f"Available samples: {len(self.test_data)}")
        self.test_iter = iter(DataLoader(test_data, conf.test_batch_size, num_workers=0, sampler=self.sampler))
        # m = 0
        # while next(self.test_iter):
        #     m+=1
        #     print("ITER", m)
        bs = conf.test_batch_size
        # self.update_model(1000)
        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            while True:
                try:
                    data = json.loads(await ws.receive_text())
                except json.decoder.JSONDecodeError as e:
                    await ws.send_text(str(e))
                    continue
                # await ws.send_text("RECEIVED")
                count = data["count"]
                if "action" in data and data["action"] == "update":
                    await ws.send_text("Update message received")
                    self.update_model(count)
                    self.logger.info("MODEL UPDATED")
                    continue
                
                self.logger.info(f"MSG: {data}")
                images, labels = next(self.test_iter)
                cc = count - images.size(0)
                while cc > 1:
                    i, l = next(self.test_iter)
                    images = torch.cat((images, i), dim=0)
                    labels = torch.cat((labels, l), dim=0)
                    cc-=i.size(0)
                self.logger.info(f"BATCH SIZE: {images.shape}")
                images = images.to(self.conf.test_device)
                tt = time.perf_counter()
                out, results = model.forward(images, self.conf, cache=False, threshold=self.confidence_threshold)#, logger=self.logger)
                tt = time.perf_counter() - tt
                confidence, pred = torch.max(out, 1)
                confidence = torch.round(torch.exp(confidence) * 100) / 100
                pred = pred.to('cpu')
                confidence = confidence.to('cpu')
                correct = torch.logical_or((pred == labels), confidence < self.confidence_threshold)
                resp = {
                    "MsgID": data["id"],
                    "corrects": correct.sum().item(), #or conf < self.confidence_threshold,
                    "confidence": str(list(np.around(confidence.cpu().detach().numpy(), decimals=2))),
                    "exit_id": str(list(results['item_exits'].cpu().detach().numpy().astype(int))),
                    "V": self.model.version,
                    "U": self.model_version,
                    "T": round(tt, 4)
                }
                
                await ws.send_json(resp)
                self.new_items += count
                self.seen_items += count
                if not self.updating and self.new_items > 1000: #and self.new_items * 5 > self.seen_items:
                    th = Thread(target=self.update_model)
                    th.start()
                    # th.join()
                    # self.test_iter = iter(DataLoader(test_data, conf.test_batch_size, True, num_workers=0))
                else:
                    self.logger.info(f"{self.new_items} & {self.seen_items} & {self.updating}")
                
            
        uvicorn.run(app, host="0.0.0.0", port=9090) 


#{"id": 1, "count": 10000, "action": "update"}
#{"id": 1, "count": 100}