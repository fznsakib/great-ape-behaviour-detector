import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

import models.resnet as resnet
import models.vgg as vgg
from utils.utils import *
from models.loss import *


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

class FusionNet(nn.Module):
    def __init__(self, spatial, temporal, num_classes):
        super(FusionNet, self).__init__()
        
        # Remove final avgpool and fc to produce an output feature size of:
        # N x 512 x 7 x 7
        self.spatial = nn.Sequential(*list(spatial.children())[:-2])
        
        # Freeze spatial layers
        # for param in self.spatial.parameters():
        #     param.requires_grad = False
        
        
        self.temporal = nn.Sequential(*list(temporal.children())[:-2])
        
        self.layer1 = nn.Sequential(
            nn.Conv3d(1024, 512, 1, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, spatial_data, temporal_data):
        x1 = self.spatial(spatial_data)
        x2 = self.temporal(temporal_data)

        y = torch.cat((x1, x2), dim=1)

        for i in range(x1.size(1)):
            y[:, (2 * i), :, :] = x1[:, i, :, :]
            y[:, (2 * i + 1), :, :] = x2[:, i, :, :]
            
        y = y.view(y.size(0), 1024, 1, 7, 7)
        cnn_out = self.layer1(y)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        out = self.fc(cnn_out)
        return out
    
class CNN:
    def __init__(self, model_name, loss, lr, regularisation, samples_per_class, num_classes, temporal_stack, device):
        super().__init__()
        
        self.lr = lr
        self.epoch = 0
        self.accuracy = 0
        self.device = device

        spatial_model = initialise_model(
            model_name=model_name, pretrained=True, num_classes=num_classes, channels=3
        )
        
        temporal_model = initialise_model(
            model_name=model_name, pretrained=False, num_classes=num_classes, channels=temporal_stack*2
        )
        
        self.model = FusionNet(spatial_model, temporal_model, num_classes)

        # Send the model to GPU
        self.model = self.model.to(device)

        self.criterion = initialise_loss(loss)
        # self.criterion = ClassBalancedFocalLoss(samples_per_class=samples_per_class, num_classes=num_classes, beta = 0.9999, gamma=2)
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=regularisation)
        self.scheduler = ReduceLROnPlateau(self.optimiser, "min", patience=1, verbose=True)

    def load_checkpoint(self, name, checkpoint_path, best=False):
        checkpoint_file_path = f"{checkpoint_path}/{name}/model"

        if best:
            checkpoint_file_path = f"{checkpoint_file_path}_best"

        if os.path.isfile(checkpoint_file_path):
            if self.device == torch.device("cuda"):
                checkpoint = torch.load(checkpoint_file_path)
            else:
                checkpoint = torch.load(checkpoint_file_path, map_location=torch.device("cpu"))

            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.epoch = checkpoint["epoch"]
            self.accuracy = checkpoint["accuracy"]

            print(
                f"==> Loaded model checkpoint {name} at epoch {self.epoch} with top1 accuracy {self.accuracy:.2f}"
            )
        else:
            print(f"==> No checkpoint at {checkpoint_file_path} -- Training model from scratch")