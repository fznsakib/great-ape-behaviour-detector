import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

import models.resnet as resnet
import models.vgg as vgg
from utils.utils import *
from models.loss import *

# Return model as required from config
def initialise_model(cnn, pretrained, num_classes, channels):
    model_initialisers = {
        "resnet18": resnet.resnet18,
        "resnet50": resnet.resnet50,
        "resnet101": resnet.resnet101,
        "resnet152": resnet.resnet152,
        "vgg16": vgg.vgg16,
        "vgg16_bn": vgg.vgg16_bn,
        "vgg19": vgg.vgg19,
        "vgg19": vgg.vgg19_bn,
    }

    model = model_initialisers[cnn](pretrained, num_classes, channels)
    return model


"""
Initial Two-Stream model using 3D Convolutional Fusion
"""
class FusionNet(nn.Module):
    def __init__(self, spatial, temporal, num_classes):
        super(FusionNet, self).__init__()

        # Remove final avgpool and fc layers to produce an output feature size of:
        # N x 512 x 7 x 7
        self.spatial = nn.Sequential(*list(spatial.children())[:-2])
        self.temporal = nn.Sequential(*list(temporal.children())[:-2])

        # 3D convolution fusion
        self.layer1 = nn.Sequential(
            nn.Conv3d(1024, 512, 1, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Fully connected classifier
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
        # Get feature map from ResNet-18 streams
        x1 = self.spatial(spatial_data)
        x2 = self.temporal(temporal_data)

        # Stack spatial and temporal feature maps
        y = torch.cat((x1, x2), dim=1)
        for i in range(x1.size(1)):
            y[:, (2 * i), :, :] = x1[:, i, :, :]
            y[:, (2 * i + 1), :, :] = x2[:, i, :, :]

        y = y.view(y.size(0), 1024, 1, 7, 7)

        # Convolutional fusion
        cnn_out = self.layer1(y)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        # Fully connected classifier
        out = self.fc(cnn_out)
        return out


"""
Final Two-Stream model using LSTM
"""
class LSTMFusionNet(nn.Module):
    def __init__(
        self,
        spatial,
        temporal,
        num_classes,
        lstm_layers,
        hidden_size,
        fc_dropout,
        lstm_dropout,
        device,
    ):
        super(LSTMFusionNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.device = device
        self.dropout = nn.Dropout(fc_dropout)

        # Remove final average pool layer
        self.spatial = nn.Sequential(*list(spatial.children())[:-1])
        self.temporal = nn.Sequential(*list(temporal.children())[:-1])

        # Initialise LSTMs for both streams
        self.lstm_spatial = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.lstm_temporal = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # Fully connected classifier
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, spatial_data, temporal_data):
        """
        Spatial forward pass
        """
        # Initialise hidden state and cell states (BPTT)
        h0_spatial = (
            torch.zeros(self.lstm_layers, spatial_data.size(0), self.hidden_size)
            .requires_grad_()
            .to(self.device)
        )
        c0_spatial = (
            torch.zeros(self.lstm_layers, spatial_data.size(0), self.hidden_size)
            .requires_grad_()
            .to(self.device)
        )

        # Reduce dimensionality of input data for ResNet-18
        batch_size, seq_length, c, h, w = spatial_data.shape
        spatial_data = spatial_data.view(batch_size * seq_length, c, h, w)
        spatial_out = self.spatial(spatial_data)

        # Revert data back to sequence for LSTM
        spatial_out = spatial_out.view(batch_size, seq_length, -1)
        spatial_out, (hn_spatial, cn_spatial) = self.lstm_spatial(
            spatial_out, (h0_spatial.detach(), c0_spatial.detach())
        )

        # Get hidden state of LSTM for final time-step
        spatial_out = spatial_out[:, -1, :]

        """
        Temporal forward pass
        """
        # Initialise hidden state and cell states (BPTT)
        h0_temporal = (
            torch.zeros(self.lstm_layers, temporal_data.size(0), self.hidden_size)
            .requires_grad_()
            .to(self.device)
        )
        c0_temporal = (
            torch.zeros(self.lstm_layers, temporal_data.size(0), self.hidden_size)
            .requires_grad_()
            .to(self.device)
        )

        # Reduce dimensionality of input data for ResNet-18
        batch_size, seq_length, c, h, w = temporal_data.shape
        temporal_data = temporal_data.view(batch_size * seq_length, c, h, w)
        temporal_out = self.temporal(temporal_data)

        # Revert data back to sequence for LSTM
        temporal_out = temporal_out.view(batch_size, seq_length, -1)
        temporal_out, (hn_temporal, cn_temporal) = self.lstm_temporal(
            temporal_out, (h0_temporal.detach(), c0_temporal.detach())
        )

        # Get hidden state of LSTM for final time-step
        temporal_out = temporal_out[:, -1, :]

        """
        Fusion
        """
        # Concatenate vectors of size 512 of spatial and temporal LSTM output
        fused_out = torch.cat((spatial_out, temporal_out), dim=1)

        # Fully connected classifier
        fused_out = F.relu(self.fc1(fused_out))
        fused_out = self.dropout(fused_out)

        return self.fc2(fused_out)


class CNN:
    def __init__(self, cfg, num_classes, device):
        super().__init__()

        # Initialise parameters
        self.lr = cfg.hyperparameters.learning_rate
        self.epoch = 0
        self.accuracy = 0
        self.device = device

        # Initialise ResNet CNNs as required by config
        spatial_model = initialise_model(
            cnn=cfg.cnn, pretrained=True, num_classes=num_classes, channels=3
        )

        temporal_model = initialise_model(
            cnn=cfg.cnn, pretrained=True, num_classes=num_classes, channels=2
        )

        # Initialise two-stream with LSTM model
        self.model = LSTMFusionNet(
            spatial=spatial_model,
            temporal=temporal_model,
            num_classes=num_classes,
            lstm_layers=cfg.lstm.layers,
            hidden_size=cfg.lstm.hidden_units,
            fc_dropout=cfg.hyperparameters.dropout,
            lstm_dropout=cfg.lstm.dropout,
            device=device,
        )

        # Send the model to GPU
        self.model = self.model.to(device)

        # Initialise loss, optimiser and learning rate scheduler
        self.criterion = initialise_loss(cfg.loss)
        self.optimiser = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=cfg.hyperparameters.sgd_momentum,
            weight_decay=cfg.hyperparameters.regularisation,
        )
        self.scheduler = ReduceLROnPlateau(self.optimiser, "min", patience=3, verbose=True)

    # Load saved model by name
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
