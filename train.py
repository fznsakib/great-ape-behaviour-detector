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
from utils.config_parser import ConfigParser

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

import random

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
        mode="validation",
        sample_interval=5,
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

    train_class_samples = train_dataset.get_no_of_samples_by_class()
    test_class_samples = test_dataset.get_no_of_samples_by_class()

    # print(train_dataset.get_videos_by_class())
    # print(test_dataset.get_videos_by_class())

    train_s = train_dataset.get_videos_by_class()
    test_s = test_dataset.get_videos_by_class()

    # print('train -> test swaps')
    # print(random.sample(train_s['camera_interaction'], k=1))
    # print(random.sample(train_s['hanging'], k=2))
    # print(random.sample(train_s['sitting_on_back'], k=1))

    # print('test -> train swaps')
    # print(random.sample(test_s['walking'], k=2))
    # print(random.sample(test_s['sitting'], k=1))
    # print(random.sample(test_s['standing'], k=1))

    # exit()

    print("==> Dataset properties")

    dataset_argument_table = [
        ["Sample Interval", cfg.dataset.sample_interval],
        ["Temporal Stack Size", cfg.dataset.temporal_stack],
        ["Activity Duration Threshold", cfg.dataset.activity_duration_threshold,],
    ]

    dataset_samples_table = [
        [
            "Train",
            train_dataset.__len__(),
            train_class_samples["camera_interaction"],
            train_class_samples["climbing_down"],
            train_class_samples["climbing_up"],
            train_class_samples["hanging"],
            train_class_samples["running"],
            train_class_samples["sitting"],
            train_class_samples["sitting_on_back"],
            train_class_samples["standing"],
            train_class_samples["walking"],
        ],
        [
            "Validation",
            test_dataset.__len__(),
            test_class_samples["camera_interaction"],
            test_class_samples["climbing_down"],
            test_class_samples["climbing_up"],
            test_class_samples["hanging"],
            test_class_samples["running"],
            test_class_samples["sitting"],
            test_class_samples["sitting_on_back"],
            test_class_samples["standing"],
            test_class_samples["walking"],
        ],
    ]

    print(tabulate(dataset_argument_table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    print(
        tabulate(
            dataset_samples_table,
            headers=[
                "Type",
                "Number of Samples",
                "camera_interaction",
                "climbing_down",
                "climbing_up",
                "hanging",
                "running",
                "sitting",
                "sitting_on_back",
                "standing",
                "walking",
            ],
            tablefmt="fancy_grid",
        )
    )

    exit()

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
    summary_writer = None
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
