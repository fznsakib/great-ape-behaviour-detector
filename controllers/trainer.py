import time
import torch
import torch.nn as nn
import numpy as np
import gc
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from statistics import mean

import utils.metrics as metrics
from utils.utils import *


class Trainer:
    def __init__(
        self,
        cnn: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        summary_writer: SummaryWriter,
        device: torch.device,
        name: str,
        save_path: Path,
        log: bool,
    ):
        self.cnn = cnn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_writer = summary_writer
        self.device = device
        self.name = name
        self.save_path = save_path
        self.log = log
        self.step = 0
        self.best_accuracy = 0
        self.validation_predictions = {}

    def train(
        self,
        epochs: int,
        start_epoch: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
    ):
        print("==> Training Stage")

        # Activate training mode
        self.cnn.model.train()

        # Train for requested number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            self.cnn.model.train()
            
            data_load_start_time = time.time()

            for i, (spatial_data, labels, metadata) in enumerate(self.train_loader):
                
                # Set gradients to zero
                self.cnn.optimiser.zero_grad()

                spatial_data = spatial_data.to(self.device)
                labels = labels.to(self.device)

                data_load_end_time = time.time()

                # Compute the forward pass of the model
                logits = self.cnn.model(spatial_data, metadata['ape_class'])

                # Compute the loss using model criterion and store it
                loss = self.cnn.criterion(logits, labels)

                # Compute the backward pass
                loss.backward()

                # Step the optimiser
                self.cnn.optimiser.step()
                
                torch.cuda.empty_cache()
                gc.collect()

                # Compute accuracy
                with torch.no_grad():
                    top1, top3 = metrics.compute_topk_accuracy(
                        logits, labels, topk=(1, 3)
                    )

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0 and self.log:
                    self.log_metrics(
                        epoch, top1, top3, loss, data_load_time, step_time,
                    )
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(
                        epoch, top1, top3, loss, data_load_time, step_time,
                    )

                self.step += 1
                data_load_start_time = time.time()

            if self.log:
                self.summary_writer.add_scalar("epoch", epoch, self.step)

            # Go into validation mode if condition met
            if ((epoch + 1) % val_frequency) == 0:
                (
                    validation_accuracy,
                    validation_loss,
                ) = self.validate()

                # Adjust LR using scheduler
                self.cnn.scheduler.step(validation_loss)

                # Switch back to train mode after validation
                self.cnn.model.train()

                # Checkpoint models if required
                if self.name:
                    is_best_model = validation_accuracy > self.best_accuracy

                    # Save model
                    if is_best_model:
                        print(f"==> Best model found with top1 accuracy {validation_accuracy:.2f}")
                        self.best_accuracy = validation_accuracy

                    print(
                        f"==> Saving model checkpoint with top1 accuracy {validation_accuracy:.2f}"
                    )
                    save_checkpoint(
                        {
                            "state_dict": self.cnn.model.state_dict(),
                            "optimiser": self.cnn.optimiser.state_dict(),
                            "epoch": epoch,
                            "accuracy": validation_accuracy,
                        },
                        is_best_model,
                        self.name,
                        self.save_path,
                    )

    def print_metrics(
        self, epoch, top1, top3, loss, data_load_time, step_time,
    ):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"top1 accuracy: {top1.item():2.2f}, "
            f"top3 accuracy: {top3.item():2.2f}, "
            f"data load time: {data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(
        self, epoch, top1, top3, loss, data_load_time, step_time,
    ):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars("top1_accuracy", {"train": top1.item()}, self.step)
        self.summary_writer.add_scalars("top3_accuracy", {"train": top3.item()}, self.step)
        self.summary_writer.add_scalars(
            "loss", {"train": float(loss.item())}, self.step
        )
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def validate(self):

        print("==> Validation Stage")

        results = {"labels": [], "logits": [], "predictions": []}
        total_loss = 0

        # Turn on evaluation mode for networks. This ensures that dropout is not applied
        # during validation and a different form of batch normalisation is used.
        self.cnn.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (spatial_data, labels, metadata) in enumerate(
                tqdm(self.test_loader, desc="Validation", leave=False, unit="batch")
            ):

                spatial_data = spatial_data.to(self.device)
                labels = labels.to(self.device)

                logits = self.cnn.model(spatial_data, metadata['ape_class'])

                loss = self.cnn.criterion(logits, labels)

                total_loss += loss.item()
                
                logits =(logits.detach().cpu()).numpy()

                # Populate dictionary with logits and labels of all samples in this batch
                for j in range(len(labels)):
                    results["labels"].append(labels[j].item())
                    results["logits"].append(logits[j].tolist())
                    results["predictions"].append(np.argmax(logits[j]).item())

        # Get accuracy by checking for correct predictions across all predictions
        top1, top3 = metrics.compute_topk_accuracy(
            torch.LongTensor(results["logits"]), torch.LongTensor(results["labels"]), topk=(1, 3)
        )

        # Get per class accuracies and sort by label value (0...9)
        class_accuracy = metrics.compute_class_accuracy(results["labels"], results["predictions"])
        class_accuracy_average = mean(class_accuracy.values())

        # Get average loss for each stream
        average_loss = total_loss / len(self.test_loader)

        # Log metrics
        if self.log:
            self.summary_writer.add_scalars(
                "loss", {"validation": average_loss}, self.step
            )
            self.summary_writer.add_scalars(
                "average_class_accuracy", {"validation": class_accuracy_average}, self.step
            )
            self.summary_writer.add_scalars("class_accuracy", {"camera_interaction": class_accuracy[0]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"climbing_down": class_accuracy[1]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"climbing_up": class_accuracy[2]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"hanging": class_accuracy[3]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"running": class_accuracy[4]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"sitting": class_accuracy[5]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"sitting_on_back": class_accuracy[6]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"standing": class_accuracy[7]}, self.step)
            self.summary_writer.add_scalars("class_accuracy", {"walking": class_accuracy[8]}, self.step)
            
            self.summary_writer.add_scalars("top1_accuracy", {"validation": top1.item()}, self.step)
            self.summary_writer.add_scalars("top3_accuracy", {"validation": top3.item()}, self.step)

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
        validation_results = [
            ["Average Loss:", f"{average_loss:.5f}"],
            ["Average Class Accuracy:", f"{class_accuracy_average:2f}"],
            ["Top1 Accuracy:", f"{top1.item():.2f}"],
            ["Top3 Accuracy:", f"{top3.item():.2f}"],
        ]

        print(tabulate(validation_results, tablefmt="fancy_grid"))

        # TODO: print per class accuracy in separate table

        return top1, average_loss
