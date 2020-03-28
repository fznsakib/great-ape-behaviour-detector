import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
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

class RNNNet(nn.Module):
    def __init__(self, model, num_classes, rnn_layers, hidden_size, dropout, device):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.num_classes = num_classes
        self.device = device
        
        self.cnn = nn.Sequential(*list(model.children())[:-1])
        self.rnn = nn.RNN(input_size = 512,
                    hidden_size = hidden_size,
                    num_layers = rnn_layers,
                    batch_first = True,
                    # nonlinearity='relu',
                    dropout = dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, data):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.rnn_layers, data.size(0), self.hidden_size).requires_grad_().to(self.device)

        batch_size, seq_length, c, h, w = data.shape
        data = data.view(batch_size * seq_length, c, h, w)
        out = self.cnn(data)
        out = out.view(batch_size, seq_length, -1)

        out, hn = self.rnn(out, (h0.detach()))
        out = out[:, -1, :]
        
        return self.fc(out)

class CNN:
    def __init__(self, model_name, loss, lr, regularisation, num_classes, temporal_stack, device):
        super().__init__()
        
        self.lr = lr
        self.epoch = 0
        self.accuracy = 0
        self.device = device

        # spatial_model = initialise_model(
        #     model_name=model_name, pretrained=True, num_classes=num_classes, channels=3
        # )
        
        temporal_model = initialise_model(
            model_name=model_name, pretrained=False, num_classes=num_classes, channels=2#channels=temporal_stack*2
        )
        
        self.model = RNNNet(temporal_model, num_classes, 2, 512, 0.2, device)

        # Send the model to GPU
        self.model = self.model.to(device)

        self.criterion = initialise_loss(loss)
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