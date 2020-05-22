import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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


# Compute the given top k accuracy of predictions made by the network
def compute_topk_accuracy(output, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, predictions = output.topk(maxk, 1, True, True)
    predictions = predictions.t().cpu()
    labels = labels.cpu()
    correct = predictions.eq(labels.view(1, -1).expand_as(predictions))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_class_accuracy(labels, predictions):

    class_accuracies = {}

    for i, label in enumerate(labels):
        if label not in class_accuracies.keys():
            class_accuracies[label] = []

        # Append 1 to label list if right prediction, otherwise append 0
        if label == predictions[i]:
            class_accuracies[label].append(1)
        else:
            class_accuracies[label].append(0)

    # Each label now has a list of 1s and 0s representing right/wrong predictions
    # Sum the list and divide by number of predictions to get the accuracy of the class
    for label in class_accuracies.keys():
        accuracy_count = class_accuracies[label]
        class_accuracies[label] = sum(accuracy_count) / len(accuracy_count)

    return class_accuracies


# Compute confusion matrix from labels and predictions
def compute_confusion_matrix(predictions_dict, classes, output_path):

    labels = []
    predictions = []
    for video in predictions_dict.keys():
        for annotation in predictions_dict[video]:
            labels.append(annotation["label"])
            predictions.append(annotation["prediction"])

    existing_classes = []

    for i in range(0, len(classes)):
        if i in set(labels):
            existing_classes.append(classes[i])

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)
    np.set_printoptions(precision=2)

    # Plot normalised confusion matrix
    plt.figure(figsize=(20, 20))
    plot_confusion_matrix(
        cnf_matrix, classes=existing_classes, normalise=True, title="Normalised confusion matrix"
    )

    plt.savefig(f"{output_path}/confusion_matrix.png", bbox_inches="tight")


# Plots the confusion matrix
def plot_confusion_matrix(
    cm, classes, normalise=False, title="Confusion matrix", cmap=plt.cm.Blues
):

    if normalise:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=30, pad=30)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = ".2f" if normalise else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=20,
        )

    plt.tight_layout()
    plt.autoscale()
    plt.ylabel("True label", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
