import models.resnet as resnet
import models.vgg as vgg

def initialise_model(model_name, pretrained, num_classes, channels):
    model_initialisers = {
        "resnet18": resnet.resnet18,
        "resnet50": resnet.resnet50,
        "resnet101": resnet.resnet101,
        "resnet152": resnet.resnet152,
        "vgg16": vgg.vgg16,
        "vgg16_bn": vgg.vgg16_bn,
        "vgg19": vgg.vgg19,
        "vgg19": vgg.vgg19_bn
    }

    model = model_initialisers[model_name](pretrained, num_classes, channels)
    return model