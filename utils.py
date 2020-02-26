import numpy as np
import torch
from typing import Union, NamedTuple

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int
    
# Compute the fusion logits across the spatial and temporal stream to perform average fusion
def average_fusion(spatial_logits, temporal_logits):
    spatial_logits = spatial_logits.cpu().detach().numpy()
    temporal_logits = temporal_logits.cpu().detach().numpy()
    fusion_logits = np.mean(np.array([spatial_logits, temporal_logits]), axis=0)

    return fusion_logits

# Compute the top1 accuracy of predictions made by the network
def compute_accuracy(labels, predictions):
    labels = labels.cpu()
    predictions = torch.from_numpy(predictions)
    assert len(labels) == len(predictions)
    return float((labels == predictions).sum()) / len(labels)

def compute_class_accuracy():
    return 0

# TODO: Compute top5 accuracy