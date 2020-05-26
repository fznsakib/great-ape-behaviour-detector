import torch
import torchvision
import torch.utils.data
import random

"""
A PyTorch Dataloader compatible sampler which ensures that every batch 
contains the same number of samples for each class.
"""
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.max_class_samples = 0

        # Save every index for every class
        for index in range(0, len(dataset)):
            label = self._get_label(dataset, index)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(index)

        # Obtain highest number of samples for a class
        for label_class in self.dataset.keys():
            if len(self.dataset[label_class]) > self.max_class_samples:
                self.max_class_samples = len(self.dataset[label_class])

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.max_class_samples:
                self.dataset[label].append(random.choice(self.dataset[label]))

        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        # Sample class until max_class_samples is met
        # Iterate through to sample classes one by one to ensure all classes sampled equally
        while self.indices[self.currentkey] < self.max_class_samples - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)

        # Reset indices at end of epoch when all clases have been sampled
        self.indices = [-1] * len(self.keys)

    # Return target label for sample
    def _get_label(self, dataset, index, labels=None):
        return self.labels[index].item()

    # Return total length of dataset when sampled equally
    # max_class_samples * number of classes
    def __len__(self):
        return self.max_class_samples * len(self.keys)
