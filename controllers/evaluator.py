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
        results = {"labels": [], "logits": [], "predictions": []}

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
                
                # Collect reulsts for metrics
                results["labels"].append(label.item())
                results["logits"].append(logits.tolist())
                results["predictions"].append(prediction)

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
                
        # Get accuracy by checking for correct predictions across all predictions
        top1, top3 = metrics.compute_topk_accuracy(
            torch.LongTensor(results["logits"]), torch.LongTensor(results["labels"]), topk=(1, 3)
        )

        # Get per class accuracies and sort by label value (0...9)
        class_accuracy = metrics.compute_class_accuracy(results["labels"], results["predictions"])
        class_accuracy_average = mean(class_accuracy.values())
        
        print("==> Per Class Results")
        per_class_results = [
            [
                "Class",
                "camera_interaction",
                "climbing_down",
                "climbing_up",
                "hanging",
                "running",
                "sitting",
                "sitting_on_back",
                "standing",
                "walking"
            ],
            [
                "Accuracy",
                f'{class_accuracy[0]:2.2f}',
                f'{class_accuracy[1]:2.2f}',
                f'{class_accuracy[2]:2.2f}',
                f'{class_accuracy[3]:2.2f}',
                f'{class_accuracy[4]:2.2f}',
                f'{class_accuracy[5]:2.2f}',
                f'{class_accuracy[6]:2.2f}',
                f'{class_accuracy[7]:2.2f}',
                f'{class_accuracy[8]:2.2f}'
            ]
        ]
        
        print(tabulate(per_class_results, tablefmt="fancy_grid"))
        
        print("==> Overall Results")
        test_results = [
            ["Average Class Accuracy:", f"{class_accuracy_average:2f}"],
            ["Top1 Accuracy:", f"{top1.item():.2f}"],
            ["Top3 Accuracy:", f"{top3.item():.2f}"],
        ]

        print(tabulate(test_results, tablefmt="fancy_grid"))

        return self.predictions
