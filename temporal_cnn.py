import torch
import torch.nn as nn
import glob


class Temporal_CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, : dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=224, width=224, channels=10)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv3d(
            in_channels=self.input_shape.channels,
            out_channels=256,
            kernel_size=(7, 7, 96),
            stride=2
        )
        self.batchNorm1 = nn.BatchNorm3d(256)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        
        self.conv2 = nn.Conv3d(
            in_channels=256,
            out_channels=512,
            kernel_size=(5, 5, 256),
            stride=2
        )
        self.batchNorm2 = nn.BatchNorm3d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)

        self.conv3 = nn.Conv3d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 512),
            stride=1
        )
        
        self.conv4 = nn.Conv3d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 512),
            stride=1
        )
        
        self.conv5 = nn.Conv3d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 512),
            stride=1
        )
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        
        # TODO: Figure out the correct number of inputs for fc1
        self.fc1 = nn.Linear(20000, 4096)
        self.initialise_layer(self.fc1)
        
        self.fc2 = nn.Linear(1024, 9, bias=False)
        self.initialise_layer(self.fc2)
        

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # First convolutional layer pass
        x = self.conv1(images)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional layer pass
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third convolutional layer pass
        x = self.conv3(x)
        x = F.relu(x)

        # Fourth convolutional layer pass
        x = self.conv4(x)
        x = F.relu(x)

        # # Flatten the output of the pooling layer so it is of shape
        # # (32, 15488), ready for fc1 to take in as input.
        # x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.conv(5)
        x = F.relu(x)
        
        # First fully connected layer pass
        x = self.fc1(x)
        x = F.relu(x)

        # Second fully connected layer pass
        x = self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if not (layer.bias is None):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)