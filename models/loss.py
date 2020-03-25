import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kornia.losses import FocalLoss

def initialise_loss(loss):
    loss_initialisers = {
        "focal": FocalLoss(alpha=0, gamma=1, reduction='mean'),
        "cross_entropy": nn.CrossEntropyLoss()
    }
    
    return loss_initialisers[loss]