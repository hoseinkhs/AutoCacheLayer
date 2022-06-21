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
        self.flops = 0
        self.cuda_time = 0 

    def batch_update(self, out, labels, results, flops):
        confidence, predicted = torch.max(out, 1)
        predicted = predicted.to('cpu')
        self.correct += (predicted == labels).sum().item()
        self.confidence += torch.exp(confidence).sum().item()
        self.total += labels.size(0)
        self.time += results["end_time"] - results["start_time"]
        self.num_batch += 1
        self.flops+=flops
        self.cuda_time += results["cuda_time"]

    
    def accuracy(self):
        return pr(self.correct, self.total)

    def mean_confidence(self):
        return pr(self.confidence, self.total)

    def mttr(self):
        return round(self.time / self.num_batch, 6) if self.num_batch else -1

    def cached_mttr(self):
        return round(self.samplewise_hit_time / self.total, 4) if self.total else -1

    def __str__(self):
        return f"Model {self.name}: \n \
        Samples: {self.total} in {self.num_batch} batches \n \
        Accuracy: {self.accuracy()} | Conf: {self.mean_confidence()} \n \
        Times: MTTR {self.mttr()} | CMTTR {self.cached_mttr()} | OvTime: {self.time:.2f}"
class ExitMeter:
    def __init__(self, name, exit_id, mm):
        self.name = name
        self.mm = mm
        self.exit_id = exit_id
        self.correct = 0
        self.cached_correct = 0
        self.hit_count = 0
        self.num_batch = 0
        self.hit_time = 0
        self.confidence = 0
        self.num_sample = 0
        self.cache_dropped = 0
        self.cache_raised = 0 

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
        try:
            confidence, predicted = torch.max(results["outputs"][i], 1)
        except:
            print(i)
            print(results["outputs"][i])
            print(results["idxs"])
        predicted = predicted.to('cpu')

        _, nc_predicted = torch.max(nc_out, 1)
        nc_predicted = nc_predicted.to('cpu')


        self.confidence += torch.exp(confidence).sum().item()
        try:
            self.correct += (predicted[hits] == labels[idxs][hits]).sum().item()
            self.cached_correct += (predicted[hits] == nc_predicted[idxs][hits]).sum().item()
            self.cache_dropped += torch.logical_and(predicted[hits] != nc_predicted[idxs][hits], labels[idxs][hits] == nc_predicted[idxs][hits]).sum().item()
            self.cache_raised +=  torch.logical_and(predicted[hits] != nc_predicted[idxs][hits], labels[idxs][hits] == predicted[hits]).sum().item()
        except:
            pass
            # print(predicted.shape)
            # print(hits.shape)
            # print(predicted[hits].shape)
            # print(idxs)
            # print(labels[idxs].shape)
            # print(labels[idxs][hits].shape)

        

        self.hit_count += num_hits
        self.hit_time += hit_time
        self.num_sample += idxs.shape[0]
        self.num_batch += 1

        self.mm.samplewise_hit_time += num_hits * hit_time

    def hit_rate(self, over_all=False):
        return pr(self.hit_count, self.mm.total if over_all else self.num_sample)
    
    def accuracy(self):
        return pr(self.correct, self.hit_count)

    def cache_accuracy(self):
        return pr(self.cached_correct, self.hit_count)
    
    def mean_confidence(self):
        return round(self.confidence / self.num_sample, 2) if self.num_sample else -1
    
    def time_ratio(self):
        return pr(self.hit_time, self.mm.time)
    def hit_time_ratio(self, meter):
        return pr(self.hit_time, meter.time)

    def accuracy_effect(self):
        return round((self.cache_raised - self.cache_dropped) / self.num_sample * 100, 2) if self.num_sample else -1

    def __str__(self):
        return f"Cache model {self.name}, Exit#{self.exit_id}: \n \
        Hit rate: {self.hit_rate()} | {self.hit_count} hits | {self.num_sample} samples | {self.num_batch} batches \n \
        Accuracy: {self.accuracy()} | CA: {self.cache_accuracy()} | Conf: {self.mean_confidence()} \n \
        Time:  {self.time_ratio()}% | OvTime: {self.hit_time}\n \
        CacheDropped: {self.cache_dropped} | CacheRaised: {self.cache_raised}"

    def __dict__(self):
        return {
            "ExitName": self.name, 
            "HitTime": self.time_ratio(),
            "HitRateOverAll": self.hit_rate(over_all=True),
            "HitRate": self.hit_rate(),
            "Accuracy": self.accuracy(),
            "CacheAccuracy": self.cache_accuracy(),
            "SamplesReached": self.num_sample,
            "BatchesReached": self.num_batch,
            "CacheDroppedAcc": self.cache_dropped,
            "CacheRaisedAcc": self.cache_raised,
            "CacheAffectAcc": self.accuracy_effect()
        }


        # {
        #     "Confidence": 1,
        #     "ExitNumber": 1.0,
        #     "ExitName": 1, 
        #     "HitTime": 1,
        #     "HitRateOverAll": 1,
        #     "HitRate": 1,
        #     "Accuracy": 1,
        #     "CacheAccuracy": 1,
        #     "SamplesReached": 1,
        #     "BatchesReached": 1
        # }