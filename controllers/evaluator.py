import time
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

import utils.metrics as metrics
from utils.utils import *


class Evaluator:
    def __init__(
        self,
        cnn: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        name: str,
    ):
        self.cnn = cnn
        self.data_loader = data_loader
        self.device = device
        self.name = name
        self.predictions = {}

    def predict(self):

        # Turn on evaluation for networkss
        self.cnn.model.eval()

        #  No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (spatial_data, temporal_data, label, metadata) in enumerate(
                tqdm(self.data_loader, desc="Prediction", leave=False, unit="sample")
            ):

                spatial_data = spatial_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                label = label.to(self.device)

                ape_id = metadata["ape_id"][0]
                start_frame = metadata["start_frame"][0]
                video = metadata["video"][0]

                logits = self.cnn.model(spatial_data, temporal_data)

                # Accumulate predictions against ground truth labels
                prediction = logits.argmax().item()

                # Insert results to dictionary
                if video not in self.predictions.keys():
                    self.predictions[video] = []

                self.predictions[video].append(
                    {
                        "ape_id": ape_id.item(),
                        "label": label.item(),
                        "prediction": prediction,
                        "start_frame": start_frame.item(),
                    }
                )

        return self.predictions
