import torch
from aux import pr
class ModelMeter:
    def __init__(self, name):
        self.name = name
        self.correct = 0
        self.num_batch = 0
        self.total = 0
        self.time = 0
        self.confidence = 0
        self.samplewise_hit_time = 0

    def batch_update(self, out, labels, results):
        confidence, predicted = torch.max(out, 1)
        predicted = predicted.to('cpu')
        self.correct += (predicted == labels).sum().item()
        self.confidence += torch.exp(confidence).sum().item()
        self.total += labels.size(0)
        self.time += results["end_time"] - results["start_time"]
        self.num_batch += 1
    
    def accuracy(self):
        return pr(self.correct, self.total)

    def mean_confidence(self):
        return pr(self.confidence, self.total)

    def mttr(self):
        return round(self.time / self.num_batch, 2) if self.num_batch else -1

    def cached_mttr(self):
        return round(self.samplewise_hit_time / self.total, 2) if self.total else -1

    def __str__(self):
        return f"Model {self.name}: \n \
        Samples: {self.total} in {self.num_batch} batches \n \
        Accuracy: {self.accuracy()} | Conf: {self.mean_confidence()} \n \
        Times: MTTR {self.mttr()} | CMTTR {self.cached_mttr()} | OvTime: {self.time:.2f}"
class ExitMeter:
    def __init__(self, exit_id, mm):
        self.mm = mm
        self.exit_id = exit_id
        self.correct = 0
        self.cached_correct = 0
        self.hit_count = 0
        self.num_batch = 0
        self.hit_time = 0
        self.confidence = 0
        self.num_sample = 0

    def batch_update(self, labels, results, nc_out):
        i = self.exit_id
        idxs = results["idxs"][i]
        if idxs.shape[0] == 0:
            return
        hits = results["hits"][i]
        hits = hits.to('cpu').bool()
        idxs = idxs.to('cpu')

        num_hits = torch.sum(hits).item()
        hit_time = (results["hit_times"][i] - results["start_time"])

        confidence, predicted = torch.max(results["outputs"][i], 1)
        predicted = predicted.to('cpu')

        _, nc_predicted = torch.max(nc_out, 1)
        nc_predicted = nc_predicted.to('cpu')


        self.confidence += torch.exp(confidence).sum().item()
        self.correct += (predicted[hits] == labels[idxs][hits]).sum().item()
        self.cached_correct += (predicted[hits] == nc_predicted[idxs][hits]).sum().item()

        self.hit_count += num_hits
        self.hit_time += hit_time
        self.num_sample += idxs.shape[0]
        self.num_batch += 1

        self.mm.samplewise_hit_time += num_hits * hit_time

    def hit_rate(self):
        return pr(self.hit_count, self.num_sample)
    
    def accuracy(self):
        return pr(self.correct, self.hit_count)

    def cache_accuracy(self):
        return pr(self.cached_correct, self.hit_count)
    
    def mean_confidence(self):
        return round(self.confidence / self.num_sample, 2) if self.num_sample else -1
    
    def time_ratio(self):
        return pr(self.hit_time, self.mm.time)

    def __str__(self):
        return f"Cache exit {self.exit_id}: \n \
        Hit rate: {self.hit_rate()} | {self.hit_count} hits | {self.num_sample} samples | {self.num_batch} batches \n \
        Accuracy: {self.accuracy()} | CA: {self.cache_accuracy()} | Conf: {self.mean_confidence()} \n \
        Time:  {self.time_ratio()}% | OvTime: {self.hit_time}"