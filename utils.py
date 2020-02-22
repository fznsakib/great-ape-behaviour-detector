import numpy as np


# Compute the fusion logits across the spatial and temporal stream to perform average fusion
def average_fusion(spatial_logits, temporal_logits):
    fusion_logits = np.mean(np.array([spatial_logits, temporal_logits]), axis=0)
    return fusion_logits

# Compute the top1 accuracy of predictions made by the network
def accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    return float((labels == predictions).sum()) / len(labels)

def class_accuracy():
    return

# TODO: Compute top5 accuracy