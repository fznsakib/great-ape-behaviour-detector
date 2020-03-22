import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def initialise_loss(loss):
    loss_initialisers = {
        "focal": FocalLoss(),
        # "cbfocal": ClassBalancedFocalLoss(),
        "cross_entropy": nn.CrossEntropyLoss()
    }
    
    return loss_initialisers[loss]
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels, reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, num_classes, beta, gamma=0):
        super(ClassBalancedFocalLoss, self).__init__()
        assert gamma >= 0
        self.beta = beta
        self.gamma = gamma
        
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.num_classes

        labels_one_hot = F.one_hot(labels, self.num_classes).float()

        weights = torch.tensor(weights, device='cuda:0').float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        return cb_loss    
        

