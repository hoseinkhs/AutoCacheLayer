import torch
import time
def threshold_confidence(x, threshold):
    x_exp = torch.exp(x)
    mx, _ = torch.max(x_exp, dim=1)
    return torch.gt(mx, threshold), mx

class CacheControl():
    def __init__(self, conf, input_shape, threshold, cache_exits, training=False, logger = None):
        device = conf.test_device if not training else conf.train_device
        self.device = device
        self.logger = logger
        self.cache_exits = cache_exits
        self.threshold = threshold
        self.conf = conf
        self.ret = torch.ones((input_shape[0], conf.num_classes)).to(device) * -1
        self.start_time = time.time()
        self.hit_times= []
        self.hits=[]
        self.idxs= [torch.arange(input_shape[0]).to(device)]
        self.outputs= []
        self.vectors= []
        self.item_exits= torch.ones(input_shape[0]).to(device) * -1
        self.exit_idx = 0
    def exit(self, out, final=False):
        current_idxs = self.idxs[-1]
        if not final:
            cache_pred = self.cache_exits[self.exit_idx](out)
            hits, mx = threshold_confidence(cache_pred, self.threshold)
            if self.logger:
                self.logger.info(f"Max cache conf: {mx}")
        else:
            self.hit_times.append(time.time())
            self.end_time = time.time()
            self.outputs.append(out)
            if out.size(0):
                self.ret[current_idxs] = out
                # self.logger.info(current_idxs, torch.ones(out.size(0)).to(self.device), self.device)
                self.item_exits[current_idxs] = torch.ones(out.size(0)).to(self.device) * -1
                self.hits.append(torch.ones(out.size(0)).to(self.device))
            return
        self.hit_times.append(time.time())
        self.outputs.append(cache_pred)
        self.hits.append(hits)
        if self.conf.shrink and hits.sum().item():
            no_hits = torch.logical_not(hits)
            hit_idxs = current_idxs[hits]
            current_idxs = current_idxs[no_hits]
            out = out[no_hits]
            self.ret[hit_idxs] = cache_pred[hits]
            # self.logger.info()
            self.item_exits[hit_idxs] = torch.ones(hit_idxs.size(0)).to(self.device) * self.exit_idx
        if not self.conf.shrink and final:
            self.ret = out
        self.idxs.append(current_idxs)
        self.exit_idx += 1
        return out, len(current_idxs) == 0
    
    def report(self):
        d = self.__dict__
        return {x: d[x] for x in d if x not in ['ret', 'conf', 'cache_exits']}