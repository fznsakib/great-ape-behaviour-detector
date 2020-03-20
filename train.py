"""""" """""" """""" """""" """""" """""" """""" """
Imports
""" """""" """""" """""" """""" """""" """""" """"""
import os
import torch
import torchvision
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from jsonargparse import ArgumentParser, ActionConfigFile
from torchvision import transforms, utils
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from pathlib import Path

"""""" """""" """""" """""" """""" """""" """""" """
Custom Library Imports
""" """""" """""" """""" """""" """""" """""" """"""
import models.spatial as spatial
import models.temporal as temporal
import models.network as network
import controllers.trainer as trainer
from dataset.dataset import GreatApeDataset
from dataset.sampler import BalancedBatchSampler
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
        device=DEVICE
    )
    
    if cfg.dataloader.sampler:
        sampler=BalancedBatchSampler(train_dataset, train_dataset.labels)
    else:
        sampler=None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.worker_count,
        sampler=sampler
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
        device=DEVICE
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.worker_count,
    )

    train_class_sample_count = train_dataset.get_no_of_samples_by_class()
    test_class_sample_count = test_dataset.get_no_of_samples_by_class()

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
            train_class_sample_count[0],
            train_class_sample_count[1],
            train_class_sample_count[2],
            train_class_sample_count[3],
            train_class_sample_count[4],
            train_class_sample_count[5],
            train_class_sample_count[6],
            train_class_sample_count[7],
            train_class_sample_count[8],
        ],
        [
            "Validation",
            test_dataset.__len__(),
            test_class_sample_count[0],
            test_class_sample_count[1],
            test_class_sample_count[2],
            test_class_sample_count[3],
            test_class_sample_count[4],
            test_class_sample_count[5],
            test_class_sample_count[6],
            test_class_sample_count[7],
            test_class_sample_count[8],
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

    # Initialise fusion CNN with spatial and temporal streams
    print("==> Initialising two-stream fusion CNN")
    cnn = network.CNN(
        model_name=cfg.model,
        loss=cfg.loss,
        lr=cfg.hyperparameters.learning_rate,
        regularisation=cfg.hyperparameters.regularisation,
        num_classes=len(classes),
        temporal_stack=cfg.dataset.temporal_stack,
        device=DEVICE
    )

    # If resuming, then load saved checkpoints
    if cfg.resume:
        cnn.load_checkpoint(cfg.name, cfg.paths.checkpoints)

    # Initialise log writing
    summary_writer = None
    if cfg.log:
        log_dir = f"{cfg.paths.logs}/{cfg.name}"
        print(f"==> Writing logs to {os.path.basename(log_dir)}")
        summary_writer = SummaryWriter(str(log_dir), flush_secs=5)
        shutil.copy('config.json', log_dir)
        

    # Initialise trainer with both CNNs
    cnn_trainer = trainer.Trainer(
        cnn,
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
    cnn_trainer.train(
        epochs=cfg.hyperparameters.epochs,
        start_epoch=cnn.epoch,
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
