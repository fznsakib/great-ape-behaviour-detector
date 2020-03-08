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
from utils import *

class Predictor:
    def __init__(
        self,
        spatial: nn.Module,
        temporal: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        name: str,
    ):
        self.spatial = spatial
        self.temporal = temporal
        self.data_loader = data_loader
        self.device = device
        self.name = name
        self.predictions = {}

    def predict(self):

        # Turn on evaluation for networkss 
        self.spatial.model.eval()
        self.temporal.model.eval()

        #  No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (spatial_data, temporal_data, label, metadata) in enumerate(
                tqdm(self.data_loader, desc="Prediction", leave=False, unit="sample")
            ):

                spatial_data = spatial_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                label = label.to(self.device)

                ape_id = metadata['ape_id'][0]
                start_frame = metadata['start_frame'][0]
                video = metadata['video'][0]

                spatial_logits = self.spatial.model(spatial_data)
                temporal_logits = self.temporal.model(temporal_data)

                # Accumulate predictions against ground truth labels
                fusion_logits = average_fusion(spatial_logits, temporal_logits)
                prediction = fusion_logits.argmax().item()

                # Insert results to dictionary
                if video not in self.predictions.keys():
                    self.predictions[video] = []

                self.predictions[video].append({
                    "ape_id": ape_id.item(),
                    "label": label.item(),
                    "prediction": prediction,
                    "start_frame": start_frame.item()
                })
        
        predictions_json = json.dumps(self.predictions)
        f = open(f"predictions.json", "w")
        f.write(predictions_json)
        f.close()

        return self.predictions