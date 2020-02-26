"""""" """""" """""" """""" """""" """""" """""" """
Imports
""" """""" """""" """""" """""" """""" """""" """"""
import os
import torch
import torchvision
import argparse
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.backends.cudnn as cudnn
from multiprocessing import cpu_count
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

"""""" """""" """""" """""" """""" """""" """""" """
Custom Library Imports
""" """""" """""" """""" """""" """""" """""" """"""
import spatial
import temporal
import trainer
from dataloader.dataset import GreatApeDataset
from utils import *

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

default_dataset_dir = f"{os.getcwd()}/mini_dataset"

parser = argparse.ArgumentParser(
    description="A spatial & temporal-based two-stream convolutional neural network for recognising great ape behaviour.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--classes", default=f'mini_dataset/classes.txt')
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--sgd-momentum", default=0.9, type=float, help="SGD momentum")
parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
parser.add_argument("--checkpoint-path", default=Path("/checkpoints"), type=Path)
parser.add_argument("--sample-interval", default=10, type=int, help="Frame interval at which samples are taken")
parser.add_argument("--optical-flow", default=5, type=int, help="Number of frames of optical flow to provide the temporal stream")
parser.add_argument("--activity-duration", default=72, type=int, help="Threshold at which a stream of activity is considered a valid sample")
parser.add_argument(
    "--spatial-dropout", default=0.5, type=float, help="Spatial dropout probability"
)
parser.add_argument(
    "--temporal-dropout", default=0.5, type=float, help="Temporal dropout probability"
)
parser.add_argument(
    "--epochs", default=50, type=int, help="Number of epochs to train the network for"
)
parser.add_argument(
    "--checkpoint-frequency", type=int, default=1, help="Save a checkpoint every N epochs"
)
parser.add_argument(
    "--val-frequency", type=int, default=1, help="Test the model on validation data every N epochs"
)
parser.add_argument(
    "--print-frequency", type=int, default=1, help="Print model metrics every N epochs"
)
parser.add_argument(
    "--log-frequency", type=int, default=1, help="Log to metrics Tensorboard every N epochs"
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

"""""" """""" """""" """""" """""" """""" """""" """
Main
""" """""" """""" """""" """""" """""" """""" """"""


def main(args):

    classes = open(args.classes).read().strip().split()
    
    # TODO: Mean flow subtraction
    train_dataset = GreatApeDataset(
        mode="train",
        sample_interval=args.sample_interval,
        no_of_optical_flow=args.optical_flow,
        activity_duration_threshold=args.activity_duration,
        video_names=f"{args.dataset_root}/splits/trainingdata.txt",
        classes=classes,
        frame_dir=f"{args.dataset_root}/frames",
        annotations_dir=f"{args.dataset_root}/annotations",
        spatial_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_count
    )
    
    test_dataset = GreatApeDataset(
        mode="validation",
        sample_interval=args.sample_interval,
        no_of_optical_flow=args.optical_flow,
        activity_duration_threshold=args.activity_duration,
        video_names=f"{args.dataset_root}/splits/validationdata.txt",
        classes=classes,
        frame_dir=f"{args.dataset_root}/frames",
        annotations_dir=f"{args.dataset_root}/annotations",
        spatial_transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_count
    )

    # Initialise CNNs for spatial and temporal streams
    spatial_model = spatial.CNN(num_classes=len(classes), device=DEVICE)
    temporal_model = temporal.CNN(num_classes=len(classes), device=DEVICE)

    # Initialise log writing
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    # Initialise trainer with both CNNs
    cnn_trainer = trainer.Trainer(
        spatial_model,
        temporal_model,
        train_loader,
        test_loader,
        summary_writer,
        DEVICE,
        args.checkpoint_frequency,
        args.checkpoint_path,
        args.log_dir,
    )

    # Begin training
    cnn_trainer.train(
        epochs=args.epochs,
        val_frequency=args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )


"""""" """""" """""" """""" """""" """""" """""" """
LOGGING FUNCTIONS
""" """""" """""" """""" """""" """""" """""" """"""

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """

    tb_log_dir_prefix = f"CNN_run_"
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


"""""" """""" """""" """""" """""" """""" """""" """
Call main()
""" """""" """""" """""" """""" """""" """""" """"""
if __name__ == "__main__":
    main(parser.parse_args())
