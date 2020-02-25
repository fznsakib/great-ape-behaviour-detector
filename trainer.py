import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
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
        checkpoint_frequency: int,
        save_path: Path,
        log_dir: str,
    ):
        self.spatial = spatial.to(device)
        self.temporal = temporal.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_writer = summary_writer
        self.device = device
        self.checkpoint_frequency = checkpoint_frequency
        self.save_path = save_path
        self.log_dir = log_dir
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
    ):

        # Activate training mode
        self.spatial.train()
        self.temporal.train()

        # Train for requested number of epochs
        for epoch in range(start_epoch, epochs):
            self.spatial.train()
            self.temporal.train()

            data_load_start_time = time.time()

            for i, (spatial_data, temporal_data, labels) in enumerate(
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
                spatial_logits = self.spatial(spatial_data)
                temporal_logits = self.temporal(temporal_data)

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
                    predictions = fusion_logits.argmax(-1)
                    accuracy = accuracy(labels, predictions)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(
                        epoch,
                        accuracy,
                        spatial_loss,
                        temporal_loss,
                        data_load_time,
                        step_time,
                    )
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(
                        epoch,
                        accuracy,
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
                validated_accuracy = self.validate()

                # Save every args.checkpoint_frequency or if this is the last epoch
                # TODO: Save network when highest accuracy is reached
                if (epoch + 1) % self.checkpoint_frequency or (epoch + 1) == epochs:
                    self.save_model(validated_accuracy)

                # Switch back to train mode after validation
                self.model.train()

    def print_metrics(
        self,
        epoch,
        accuracy,
        spatial_loss,
        temporal_loss,
        data_load_time,
        step_time,
    ):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"spatial batch loss: {spatial_loss:.5f}, "
            f"temporal batch loss: {temporal_loss:.5f}, "
            f"accuracy: {accuracy * 100:2.2f}, "
            f"data load time: {data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def log_metrics(
        self,
        epoch,
        accuracy,
        spatial_loss,
        temporal_loss,
        data_load_time,
        step_time,
    ):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
            "accuracy", {"train": accuracy}, self.step
        )
        self.summary_writer.add_scalars(
            "spatial_loss", {"train": float(spatial_loss.item())}, self.step
        )
        self.summary_writer.add_scalars(
            "temporal_loss", {"train": float(temporal_loss.item())}, self.step
        )
        self.summary_writer.add_scalar("time/data", data_load_time, self.step)
        self.summary_writer.add_scalar("time/data", step_time, self.step)

    def save_model(self, accuracy):
        print(f"Saving model to {self.save_path} with accuracy of {accuracy*100:2.2f}")
        torch.save(
            {"model": self.model.state_dict(), "accuracy": accuracy}, self.save_path
        )

    def validate(self):
        results = {"predictions": [], "labels": []}
        total_spatial_loss = 0
        total_temporal_loss = 0

        # Turn on evaluation mode for network. This ensures that dropout is not applied
        # during validation and a different form of batch normalisation is used.
        self.spatial.eval()
        self.temporal.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (spatial_data, temporal_data, labels) in enumerate(
                self.test_loader
            ):
                spatial_data = spatial_data.to(self.device)
                temporal_data = temporal_data.to(self.device)
                labels = labels.to(self.device)
                
                spatial_logits = self.spatial(spatial_data)
                temporal_logits = self.temporal(temporal_data)
                
                spatial_loss = self.spatial.criterion(spatial_logits, labels)
                temporal_loss = self.temporal.criterion(temporal_logits, labels)
                
                total_spatial_loss += spatial_loss.item()
                total_temporal_loss += temporal_loss.item()

                # Accumulate predictions against ground truth labels
                fusion_logits = average_fusion(spatial_logits, temporal_logits)
                predictions = fusion_logits.argmax(dim=-1).cpu().numpy()
                results["predictions"].extend(list(predictions))
                results["labels"].extend(list(labels.cpu().numpy()))

        # Get accuracy by checking for correct predictions across all predictions
        accuracy = accuracy()

        # Get per class accuracies and sort by label value (0...9)
        class_accuracy = class_accuracy()

        # Get average loss for each stream
        average_spatial_loss = total_spatial_loss / len(self.test_loader)
        average_temporal_loss = total_temporal_loss / len(self.test_loader)

        # Log metrics
        self.summary_writer.add_scalars(
            "accuracy", {"validation": accuracy}, self.step
        )
        self.summary_writer.add_scalars("average_spatial_loss", {"validation": average_spatial_loss}, self.step)
        self.summary_writer.add_scalars("average_temporal_loss", {"validation": average_temporal_loss}, self.step)

        print(
            f"Validation results:\n"
            f"avg spatial loss: {average_spatial_loss:.5f}, "
            f"avg temporal loss: {average_temporal_loss:.5f}, "
            f"accuracy: {accuracy * 100:2.2f}\n"
            f"per class accuracies: {per_class_accuracies}"
        )
        return accuracy
