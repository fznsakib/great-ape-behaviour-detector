import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from utils import *


class Trainer:
    def __init__(
        self,
        spatial: nn.Module,
        temporal: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        summary_writer: SummaryWriter,
        device: torch.device,
        name: str,
        save_path: Path,
    ):
        self.spatial = spatial
        self.temporal = temporal
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_writer = summary_writer
        self.device = device
        self.name = name
        self.save_path = save_path
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
        self.spatial.model.train()
        self.temporal.model.train()

        # Train for requested number of epochs
        for epoch in range(start_epoch, start_epoch + epochs):
            self.spatial.model.train()
            self.temporal.model.train()

            data_load_start_time = time.time()

            for i, (spatial_data, temporal_data, labels, metadata) in enumerate(
                self.train_loader
            ):
                # Set gradients to zero
                self.spatial.optimiser.zero_grad()
                self.temporal.optimiser.zero_grad()

                spatial_data = spatial_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                labels = labels.to(self.device)

                data_load_end_time = time.time()

                # Compute the forward pass of the model
                spatial_logits = self.spatial.model(spatial_data)
                temporal_logits = self.temporal.model(temporal_data)

                # Compute the loss using model criterion and store it
                spatial_loss = self.spatial.criterion(spatial_logits, labels)
                temporal_loss = self.temporal.criterion(temporal_logits, labels)

                # Compute the backward pass
                spatial_loss.backward()
                temporal_loss.backward()

                # Step the optimiser
                self.spatial.optimiser.step()
                self.temporal.optimiser.step()

                # Compute accuracy
                with torch.no_grad():
                    fusion_logits = average_fusion(spatial_logits, temporal_logits)
                    top1, top3 = compute_topk_accuracy(
                        torch.from_numpy(fusion_logits), labels, topk=(1, 3)
                    )

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(
                        epoch,
                        top1,
                        top3,
                        spatial_loss,
                        temporal_loss,
                        data_load_time,
                        step_time,
                    )
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(
                        epoch,
                        top1,
                        top3,
                        spatial_loss,
                        temporal_loss,
                        data_load_time,
                        step_time,
                    )

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # Go into validation mode if condition met
            if ((epoch + 1) % val_frequency) == 0:
                validation_accuracy, validation_spatial_loss, validation_temporal_loss = self.validate()

                # Adjust LR using scheduler
                self.spatial.scheduler.step(validation_spatial_loss)
                self.temporal.scheduler.step(validation_temporal_loss)

                # Switch back to train mode after validation
                self.spatial.model.train()
                self.temporal.model.train()

                # Checkpoint models if required
                if self.name:
                    is_best_model = validation_accuracy > self.best_accuracy

                    # Save model
                    if is_best_model:
                        print(f'==> Best model found with top1 accuracy {validation_accuracy}')
                        self.best_accuracy = validation_accuracy

                    print(f'==> Saving model checkpoint with top1 accuracy {validation_accuracy}')
                    save_checkpoint(
                        {
                            "state_dict": self.spatial.model.state_dict(),
                            "optimiser": self.spatial.optimiser.state_dict(),
                            "epoch": epoch,
                            "accuracy": validation_accuracy,
                        },
                        {
                            "state_dict": self.temporal.model.state_dict(),
                            "optimiser": self.temporal.optimiser.state_dict(),
                            "epoch": epoch,
                            "accuracy": validation_accuracy,
                        },
                        is_best_model,
                        self.name,
                        self.save_path,
                    )

    def print_metrics(
        self, epoch, top1, top3, spatial_loss, temporal_loss, data_load_time, step_time,
    ):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"spatial batch loss: {spatial_loss:.5f}, "
            f"temporal batch loss: {temporal_loss:.5f}, "
            f"top1 accuracy: {top1.item():2.2f}, "
            f"top3 accuracy: {top3.item():2.2f}, "
            f"data load time: {data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(
        self, epoch, top1, top3, spatial_loss, temporal_loss, data_load_time, step_time,
    ):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "top1_accuracy", {"train": top1.item()}, self.step
        )
        self.summary_writer.add_scalars(
            "top3_accuracy", {"train": top3.item()}, self.step
        )
        self.summary_writer.add_scalars(
            "spatial_loss", {"train": float(spatial_loss.item())}, self.step
        )
        self.summary_writer.add_scalars(
            "temporal_loss", {"train": float(temporal_loss.item())}, self.step
        )
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)


    def validate(self):

        print("==> Validation Stage")

        results = {"labels": [], "logits": []}
        total_spatial_loss = 0
        total_temporal_loss = 0

        # Turn on evaluation mode for networks. This ensures that dropout is not applied
        # during validation and a different form of batch normalisation is used.
        self.spatial.model.eval()
        self.temporal.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (spatial_data, temporal_data, labels, metadata) in enumerate(
                tqdm(self.test_loader, desc="Validation", leave=False, unit="batch")
            ):

                spatial_data = spatial_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                labels = labels.to(self.device)

                spatial_logits = self.spatial.model(spatial_data)
                temporal_logits = self.temporal.model(temporal_data)

                spatial_loss = self.spatial.criterion(spatial_logits, labels)
                temporal_loss = self.temporal.criterion(temporal_logits, labels)

                total_spatial_loss += spatial_loss.item()
                total_temporal_loss += temporal_loss.item()

                # Accumulate predictions against ground truth labels
                fusion_logits = average_fusion(spatial_logits, temporal_logits)

                # Populate dictionary with logits and labels of all samples in this batch
                for i in range(len(labels)):
                    results['labels'].append(labels[i].item())
                    results['logits'].append(fusion_logits[i].tolist())


        # Get accuracy by checking for correct predictions across all predictions
        top1, top3 = compute_topk_accuracy(
            torch.LongTensor(results['logits']), torch.LongTensor(results['labels']), topk=(1, 3)
        )

        # Get per class accuracies and sort by label value (0...9)
        per_class_accuracy = compute_class_accuracy()

        # Get average loss for each stream
        average_spatial_loss = total_spatial_loss / len(self.test_loader)
        average_temporal_loss = total_temporal_loss / len(self.test_loader)

        # Log metrics
        self.summary_writer.add_scalars(
            "top1_accuracy", {"validation": top1.item()}, self.step
        )
        self.summary_writer.add_scalars(
            "top3_accuracy", {"validation": top3.item()}, self.step
        )
        self.summary_writer.add_scalars(
            "spatial_loss", {"validation": average_spatial_loss}, self.step
        )
        self.summary_writer.add_scalars(
            "temporal_loss", {"validation": average_temporal_loss}, self.step
        )

        print("==> Results")
        validation_results = [
            ["Average Spatial Loss:", f"{average_spatial_loss:.5f}"],
            ["Average Temporal Loss:", f"{average_temporal_loss:.5f}"],
            ["Top1 Accuracy:", f"{top1.item()}"],
            ["Top3 Accuracy:", f"{top3.item()}"],
        ]

        print(tabulate(validation_results, tablefmt="fancy_grid"))

        # TODO: print per class accuracy in separate table

        return top1, average_spatial_loss, average_temporal_loss
