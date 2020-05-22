import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kornia.losses import FocalLoss

weights = None

# Return loss function as required by config
def initialise_loss(loss):
    if loss.function == "focal":
        return FocalLoss(alpha=loss.focal.alpha, gamma=loss.focal.gamma, reduction="mean")

    if loss.function == "cross_entropy":
        if loss.cross_entropy.weighted:
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()


# Compute weights for weighted cross entropy if required
def initialise_ce_weights(class_sample_count):
    largest_class_size = max(class_sample_count)
    weights = torch.empty(len(class_sample_count), dtype=torch.float)
    for i in range(0, len(class_sample_count)):
        weights[i] = largest_class_size / class_sample_count[i]
