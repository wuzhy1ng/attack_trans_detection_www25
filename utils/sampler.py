import random

from torch.utils.data import BatchSampler


class DeduplicateBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super(DeduplicateBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.sampler = sampler

    def __iter__(self):
        batch, vis_labels = list(), set()
        for idx in self.sampler:
            if random.random() < 0.5:  # random shuffle (approximately)
                continue
            try:
                data = self.sampler.data_source[idx]
            except:
                continue
            label = eval(data.label)
            label = label.get('func_name')
            if label in vis_labels:
                continue
            batch.append(idx)
            vis_labels.add(label)
            if len(batch) < self.batch_size:
                continue
            yield batch
            batch, vis_labels = list(), set()

        if len(batch) > 0 and not self.drop_last:
            yield batch
