import math
import torch
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo
from torchvision import models

# Download links for pre-trained VGG weights
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


"""
Network Customisation
"""
# Transform the original 3 channel weight to the number channels requested
def cross_modality_pretrain(conv1_weight, channels):
    # Accumulate conv1 weights across the 3 channels
    S = 0
    for i in range(3):
        S += conv1_weight[:, i, :, :]

    # Get average of weights
    avg = S / 3.0
    new_conv1_weight = torch.FloatTensor(64, channels, 3, 3)

    # Assign average weight to each of the channels in conv1
    for i in range(channels):
        new_conv1_weight[:, i, :, :] = avg.data

    return new_conv1_weight


# Adapt pretrained weights of conv1 to created model according to number of channels
def weight_transform(model_dict, pretrain_dict, channels):
    weight_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    w3 = pretrain_dict["features.0.weight"]

    if channels == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3, channels)

    # Assign pretrained weights to model
    weight_dict["features.0.weight"] = wt
    model_dict.update(weight_dict)
    return model_dict


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def add_classifier(model, num_classes):
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
    )

    return model


def customise_initial_layer(model, model_name, channels):
    model.features[0] = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=True)

    pretrain_dict = model_zoo.load_url(model_urls[model_name])
    new_model_dict = weight_transform(model.state_dict(), pretrain_dict, channels)
    model.state_dict()["features.0.weight"] = new_model_dict["features.0.weight"]
    return model


"""
VGG Definitions
"""
def vgg16(pretrained, num_classes, channels):
    model = models.vgg16(pretrained=pretrained)

    if channels != 3:
        model = customise_initial_layer(model, "vgg16", channels)

    model = freeze_weights(model)
    model = add_classifier(model, num_classes)

    return model


def vgg16_bn(pretrained, num_classes, channels):
    model = models.vgg16_bn(pretrained=pretrained)

    if channels != 3:
        model = customise_initial_layer(model, "vgg16_bn", channels)

    model = freeze_weights(model)
    model = add_classifier(model, num_classes)

    return model


def vgg19(pretrained, num_classes, channels):
    model = models.vgg16(pretrained=pretrained)

    if channels != 3:
        model = customise_initial_layer(model, "vgg19", channels)

    model = freeze_weights(model)
    model = add_classifier(model, num_classes)

    return model


def vgg19_bn(pretrained, num_classes, channels):
    model = models.vgg16(pretrained=pretrained)

    if channels != 3:
        model = customise_initial_layer(model, "vgg19_bn", channels)

    model = freeze_weights(model)
    model = add_classifier(model, num_classes)

    return model
