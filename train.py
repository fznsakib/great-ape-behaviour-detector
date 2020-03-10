"""""" """""" """""" """""" """""" """""" """""" """
Imports
""" """""" """""" """""" """""" """""" """""" """"""
import os
import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from jsonargparse import ArgumentParser, ActionConfigFile
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from pathlib import Path

"""""" """""" """""" """""" """""" """""" """""" """
Custom Library Imports
""" """""" """""" """""" """""" """""" """""" """"""
import models.spatial as spatial
import models.temporal as temporal
import controllers.trainer as trainer
from dataset.dataset import GreatApeDataset
from utils.utils import *
from config_parser import ConfigParser

"""""" """""" """""" """""" """""" """""" """""" """
GPU Initialisation
""" """""" """""" """""" """""" """""" """""" """"""
torch.backends.cudnn.benchmark = True

# Check if GPU available, and use if so. Otherwise, use CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

"""""" """""" """""" """""" """""" """""" """""" """
Argument Parser
""" """""" """""" """""" """""" """""" """""" """"""

default_data_path = Path(f"{os.getcwd()}/../scratch/data")
default_checkpoints_path = Path(f"{os.getcwd()}/../scratch/checkpoints")
default_logs_path = Path(f"{os.getcwd()}/../scratch/logs")
default_classes_path = Path(f"{default_data_path}/classes.txt")

parser = argparse.ArgumentParser(
    description="A spatial & temporal-based two-stream convolutional neural network for recognising great ape behaviour.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


"""""" """""" """""" """""" """""" """""" """""" """
Main
""" """""" """""" """""" """""" """""" """""" """"""


def main(cfg):

    classes = open(cfg.paths.classes).read().strip().split()

    print("==> Initialising training dataset")

    train_dataset = GreatApeDataset(
        mode=cfg.mode,
        sample_interval=cfg.dataset.sample_interval,
        temporal_stack=cfg.dataset.temporal_stack,
        activity_duration_threshold=cfg.dataset.activity_duration_threshold,
        video_names=f"{cfg.paths.splits}/trainingdata.txt",
        classes=classes,
        frame_dir=cfg.paths.frames,
        annotations_dir=cfg.paths.annotations,
        spatial_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ]
        ),
        temporal_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.worker_count,
    )

    print("==> Initialising validation dataset")

    test_dataset = GreatApeDataset(
        mode=cfg.mode,
        sample_interval=cfg.dataset.sample_interval,
        temporal_stack=cfg.dataset.temporal_stack,
        activity_duration_threshold=cfg.dataset.activity_duration_threshold,
        video_names=f"{cfg.paths.splits}/validationdata.txt",
        classes=classes,
        frame_dir=cfg.paths.frames,
        annotations_dir=cfg.paths.annotations,
        spatial_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ]
        ),
        temporal_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.worker_count,
    )

    print("==> Dataset properties")

    dataset_argument_table = [
        ["Sample Interval", cfg.dataset.sample_interval],
        ["Temporal Stack Size", cfg.dataset.temporal_stack],
        ["Activity Duration Threshold", cfg.dataset.activity_duration_threshold,],
    ]

    dataset_table = [
        ["Train", train_dataset.__len__()],
        ["Validation", test_dataset.__len__()],
    ]

    print(tabulate(dataset_argument_table, headers=["Parameter", "Value"], tablefmt="fancy_grid",))
    print(
        tabulate(
            dataset_table, headers=["Dataset Type", "Number of Samples"], tablefmt="fancy_grid",
        )
    )

    # Initialise CNNs for spatial and temporal streams
    spatial_model = spatial.CNN(
        model_name=cfg.model,
        lr=cfg.hyperparameters.learning_rate,
        num_classes=len(classes),
        channels=3,
        device=DEVICE,
    )
    temporal_model = temporal.CNN(
        model_name=cfg.model,
        lr=cfg.hyperparameters.learning_rate,
        num_classes=len(classes),
        channels=cfg.dataset.temporal_stack * 2,
        device=DEVICE,
    )

    # If resuming, then load saved checkpoints
    if cfg.resume:
        spatial_model.load_checkpoint(cfg.name, cfg.paths.checkpoints)
        temporal_model.load_checkpoint(cfg.name, cfg.paths.checkpoints)

    # Initialise log writing
    if cfg.log:
        log_dir = f"{cfg.paths.logs}/{cfg.name}"
        print(f"==> Writing logs to {os.path.basename(log_dir)}")
        summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    # Initialise trainer with both CNNs
    cnn_trainer = trainer.Trainer(
        spatial_model,
        temporal_model,
        train_loader,
        test_loader,
        summary_writer,
        DEVICE,
        cfg.name,
        cfg.paths.checkpoints,
        cfg.log,
    )

    # Begin training
    print("==> Begin training")
    start_epoch = max(spatial_model.start_epoch, temporal_model.start_epoch)
    cnn_trainer.train(
        epochs=cfg.hyperparameters.epochs,
        start_epoch=start_epoch,
        val_frequency=cfg.frequencies.validation,
        print_frequency=cfg.frequencies.print,
        log_frequency=cfg.frequencies.log,
    )


"""""" """""" """""" """""" """""" """""" """""" """
Call main()
""" """""" """""" """""" """""" """""" """""" """"""
if __name__ == "__main__":
    cfg = ConfigParser().config
    main(cfg)