import os
import torch
import numpy as np
from typing import NamedTuple
    
# Compute the fusion logits across the spatial and temporal stream to perform average fusion
def average_fusion(spatial_logits, temporal_logits):
    spatial_logits = spatial_logits.cpu().detach().numpy()
    temporal_logits = temporal_logits.cpu().detach().numpy()
    fusion_logits = np.mean(np.array([spatial_logits, temporal_logits]), axis=0)

    return fusion_logits

# Compute the top1 accuracy of predictions made by the network
def compute_accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    
    correct_predictions = 0
    for i, prediction in enumerate(predictions):
        if prediction == labels[i]:
            correct_predictions += 1

    return float(correct_predictions) / len(predictions)

def compute_topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    target = target.cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_class_accuracy():
    return 0

def save_checkpoint(spatial_state, temporal_state, is_best_model, name, save_path):    

    checkpoint_path = f'{save_path}/{name}'
    
    if not os.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    torch.save(spatial_state, f'{checkpoint_path}/spatial')
    torch.save(temporal_state, f'{checkpoint_path}/temporal')
    
    if is_best_model:
        shutil.copyfile(f'{checkpoint_path}/spatial', f'{checkpoint_path}/spatial_best')
        shutil.copyfile(f'{checkpoint_path}/temporal', f'{checkpoint_path}/temporal_best')