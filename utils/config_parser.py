import os
import json
from pathlib import Path
from jsonargparse import ArgumentParser, ActionConfigFile, ActionPath
from multiprocessing import cpu_count

from utils.utils import *


class ConfigParser:
    def __init__(self):

        # Initialise parser
        self.parser = ArgumentParser(
            description="A spatial & temporal-based two-stream convolutional neural network for recognising great ape behaviour."
        )

        # Take in config
        self.parser.add_argument("--config", action=ActionConfigFile)

        # Add all arguments to parser
        self.add_general_arguments()
        self.add_hyperparameter_arguments()
        self.add_path_arguments()
        self.add_dataset_arguments()
        self.add_dataloader_arguments()
        self.add_frequency_arguments()

        # Create name
        self.config = self.parser.parse_args()

        if not self.config.name:
            if self.config.mode == "test":
                print("Please specify model name in order to evaluate")
                exit()
            self.config.name = "test"

        self.config.dataloader.worker_count = cpu_count()

        if self.config.mode == "test":
            self.config.dataloader.batch_size = 1
            self.config.dataloader.shuffle = False
            self.config.dataset.sample_interval = 5

        self.config.paths.annotations = str(self.config.paths.annotations)
        self.config.paths.checkpoints = str(self.config.paths.checkpoints)
        self.config.paths.classes = str(self.config.paths.classes)
        self.config.paths.frames = str(self.config.paths.frames)
        self.config.paths.logs = str(self.config.paths.logs)
        self.config.paths.output = str(self.config.paths.output)
        self.config.paths.splits = str(self.config.paths.splits)

    def add_general_arguments(self):
        self.parser.add_argument(
            "--name", default="", type=str, help="Toggle model checkpointing by providing name",
        )
        self.parser.add_argument("--mode", default="", type=str, help="Choose model functionality")
        self.parser.add_argument(
            "--resume", default=False, type=bool, help="Load and resume model for training",
        )
        self.parser.add_argument(
            "--best",
            default=False,
            type=bool,
            help="Load the checkpoint with the best accuracy for the model",
        )

        self.parser.add_argument(
            "--model", default="resnet18", type=str, help="Type of pretrained model to load",
        )
        self.parser.add_argument(
            "--loss", default="cross_entropy", type=str, help="Loss function to use",
        )

        self.parser.add_argument(
            "--bucket",
            default="faizaanbucket",
            type=str,
            help="AWS S3 bucket name to upload output to.",
        )

        self.parser.add_argument(
            "--log", default=True, type=bool, help="Log metrics for this train run.",
        )

    def add_hyperparameter_arguments(self):
        self.parser.add_argument(
            "--hyperparameters.epochs",
            default=50,
            type=int,
            help="Number of epochs to train the network for",
        )
        self.parser.add_argument(
            "--hyperparameters.learning_rate", default=0.001, type=float, help="Learning rate",
        )
        self.parser.add_argument(
            "--hyperparameters.sgd_momentum", default=0.9, type=float, help="SGD momentum",
        )

    def add_path_arguments(self):
        self.parser.add_argument(
            "--paths.annotations",
            type=Path,
            help="Path to root of dataset",
            action=ActionPath(mode="drw"),
        )
        self.parser.add_argument(
            "--paths.frames", type=Path, help="Path to frames", action=ActionPath(mode="drw"),
        )
        self.parser.add_argument(
            "--paths.logs",
            type=Path,
            help="Path to where logs will be saved",
            action=ActionPath(mode="drw"),
        )
        self.parser.add_argument(
            "--paths.classes", type=Path, help="Path to classes.txt", action=ActionPath(mode="fr"),
        )
        self.parser.add_argument(
            "--paths.splits", type=Path, help="Path to data splits", action=ActionPath(mode="drw"),
        )
        self.parser.add_argument(
            "--paths.output", type=Path, help="Path to model output", action=ActionPath(mode="drw"),
        )
        self.parser.add_argument(
            "--paths.checkpoints",
            type=Path,
            help="Path to root of saved model checkpoints",
            action=ActionPath(mode="drw"),
        )

    def add_dataset_arguments(self):
        self.parser.add_argument(
            "--dataset.sample_interval",
            default=10,
            type=int,
            help="Frame interval at which samples are taken",
        )
        self.parser.add_argument(
            "--dataset.temporal_stack",
            default=5,
            type=int,
            help="Number of frames of optical flow to provide the temporal stream",
        )
        self.parser.add_argument(
            "--dataset.activity_duration_threshold",
            default=72,
            type=int,
            help="Threshold at which a stream of activity is considered a valid sample",
        )

    def add_dataloader_arguments(self):
        self.parser.add_argument("--dataloader.batch_size", default=32, type=int, help="Batch size")
        self.parser.add_argument(
            "--dataloader.shuffle", default=False, type=bool, help="Shuffle data"
        )
        self.parser.add_argument(
            "--dataloader.worker_count", default=1, type=int, help="Number of workers to load data",
        )
        self.parser.add_argument(
            "--dataloader.sampler", default=False, type=bool, help="Use balanced batch sampler for equal number of samples across classes",
        )

    def add_frequency_arguments(self):
        self.parser.add_argument(
            "--frequencies.validation",
            type=int,
            default=1,
            help="Test the model on validation data every N epochs",
        )
        self.parser.add_argument(
            "--frequencies.log",
            type=int,
            default=1,
            help="Log to metrics Tensorboard every N epochs",
        )
        self.parser.add_argument(
            "--frequencies.print", type=int, default=1, help="Print model metrics every N epochs",
        )
