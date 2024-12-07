import torch
import random


class MemoryBank:
    def __init__(self, feature_dim, num_classes, bank_size_per_class):
        self.num_classes = num_classes
        self.bank_size_per_class = bank_size_per_class
        self.memory_bank = {cls: torch.zeros((bank_size_per_class, feature_dim)) for cls in range(num_classes)}
        self.ptrs = {cls: 0 for cls in range(num_classes)}

    def update(self, features, labels):
        for feat, label in zip(features, labels):
            feat = feat.detach()
            label = label.item()
            # 随机选择要替换的索引
            rand_idx = random.randint(0, self.bank_size_per_class - 1)
            self.memory_bank[label][rand_idx] = feat

    def sample(self, batch_size_per_class):
        samples = []
        labels = []
        for cls in range(self.num_classes):
            idxs = torch.randperm(self.bank_size_per_class)[:batch_size_per_class]
            samples.append(self.memory_bank[cls][idxs])
            labels.append(torch.full((batch_size_per_class,), cls, dtype=torch.int8))
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)
