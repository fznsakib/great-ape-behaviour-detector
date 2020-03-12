import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models.network as network
from utils.utils import *
from models.loss import FocalLoss


class CNN:
    def __init__(self, model_name, lr, num_classes, channels, device):
        super().__init__()

        self.lr = lr
        self.start_epoch = 0
        self.accuracy = 0
        self.device = device

        print("==> Initialising spatial CNN model")

        self.model = network.initialise_model(
            model_name=model_name, pretrained=True, num_classes=num_classes, channels=channels
        )

        # Send the model to GPU
        self.model = self.model.to(device)

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss()
        self.optimiser = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimiser, "min", patience=1, verbose=True)

    def load_checkpoint(self, name, checkpoint_path, best=False):
        checkpoint_file_path = f"{checkpoint_path}/{name}/spatial"

        if best:
            checkpoint_file_path = f"{checkpoint_file_path}_best"

        if os.path.isfile(checkpoint_file_path):
            if self.device == torch.device("cuda"):
                checkpoint = torch.load(checkpoint_file_path)
            else:
                checkpoint = torch.load(checkpoint_file_path, map_location=torch.device("cpu"))

            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.start_epoch = checkpoint["epoch"]
            self.accuracy = checkpoint["accuracy"]

            print(
                f"==> Loaded spatial model checkpoint {name} at epoch {self.start_epoch} with top1 accuracy {self.accuracy:.2f}"
            )
        else:
            print(f"==> No checkpoint at {checkpoint_file_path} -- Training model from scratch")
