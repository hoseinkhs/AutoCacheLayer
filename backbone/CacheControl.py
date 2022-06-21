import torch
import time
def threshold_confidence(x, threshold):
    x_exp = torch.exp(x)
    mx, _ = torch.max(x_exp, dim=1)
    return torch.gt(mx, threshold), mx

class CacheControl():
    def __init__(self, conf, input_shape, threshold, cache_exits, training=False, logger = None):
        device = conf.test_device if not training else conf.train_device
        self.num_exits = len(cache_exits) + 1
        self.input_shape = input_shape
        self.device = device
        self.logger = logger
        self.cache_exits = cache_exits
        self.threshold = threshold
        self.conf = conf
        self.ret = torch.ones((input_shape[0], conf.num_classes)).to(device) * -1
        self.start_time = time.time()
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.hit_times= [0 for i in range(self.num_exits)]
        self.hits= [torch.ones(0) for i in range(self.num_exits)]
        self.idxs= [torch.arange(input_shape[0]).to(device)]
        self.outputs= [torch.ones(0) for i in range(self.num_exits)]
        self.vectors= []
        self.item_exits = torch.ones(input_shape[0]).to(device) * -1
        self.exit_idx = 0
        self.cuda_time = 0
        self.starter.record()

    def all_resolved(self):
        if not self.exit_idx:
            return False
        x = self.hits[0]
        for i in range(1, self.exit_idx):
            x = torch.logical_or(x, self.hits[i])
        return x.sum().item() == self.input_shape[0]
    def record_timing(self):
        self.end_time = time.time()  
        self.ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        self.cuda_time = self.starter.elapsed_time(self.ender)

    def exit(self, out, final=False):
        current_idxs = self.idxs[-1]
        if not final:
            cache_pred = self.cache_exits[self.exit_idx](out)
            hits, mx = threshold_confidence(cache_pred, self.threshold)
            # print("Cache result:", hits, mx)
            if self.logger:
                self.logger.info(f"Max cache conf: {mx}")
        else:
            self.hit_times[self.exit_idx] = time.time()
            self.outputs[self.exit_idx] = out
            self.record_timing()
            if out.size(0):
                # print(current_idxs, out.shape)
                self.ret[current_idxs] = out
                # self.logger.info(current_idxs, torch.ones(out.size(0)).to(self.device), self.device)
                self.item_exits[current_idxs] = torch.ones(out.size(0)).to(self.device) * -1
                self.hits[self.exit_idx] = torch.ones(out.size(0)).to(self.device)
                # else:
                #     self.hits.append(torch.ones(0))
            return
        self.hit_times[self.exit_idx] = time.time()
        self.outputs[self.exit_idx] = cache_pred
        self.hits[self.exit_idx] = hits
        if self.conf.shrink and hits.sum().item():
            no_hits = torch.logical_not(hits)
            hit_idxs = current_idxs[hits]
            current_idxs = current_idxs[no_hits]
            out = out[no_hits]
            self.ret[hit_idxs] = cache_pred[hits]
            self.item_exits[hit_idxs] = torch.ones(hit_idxs.size(0)).to(self.device) * self.exit_idx
            
        if not self.conf.shrink and final:
            self.ret = out
        should_exit = len(current_idxs) == 0 or (self.conf.exit_on_all_resolved and self.all_resolved())
        if should_exit:
            self.record_timing()
        else:
            self.idxs.append(current_idxs)
            self.exit_idx += 1
        return out, should_exit
    
    def report(self):
        d = self.__dict__
        return {x: d[x] for x in d if x not in ['ret', 'conf', 'cache_exits']}