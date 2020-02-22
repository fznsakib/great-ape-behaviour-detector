"""""" """""" """""" """""" """""" """""" """""" """
Imports
""" """""" """""" """""" """""" """""" """""" """"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torch.utils.data import Dataset, DataLoader

"""""" """""" """""" """""" """""" """""" """""" """
Custom Library Imports
""" """""" """""" """""" """""" """""" """""" """"""
import spatial
import temporal
import trainer

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

default_dataset_dir = f"{os.getcwd()}/mini_dataset/"

parser = argparse.ArgumentParser(
    description="A spatial & temporal-based two-stream convolutional neural network for recognising great ape behaviour.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--sgd-momentum", default=0.9, type=float, help="SGD momentum")
parser.add_argument("--spatial-dropout", default=0.5, type=float, help="Spatial dropout probability")
parser.add_argument("--temporal-dropout", default=0.5, type=float, help="Temporal dropout probability")
parser.add_argument("--checkpoint-path", default=Path("/weights"), type=Path)
parser.add_argument(
    "--epochs", default=50, type=int, help="Number of epochs to train the network for"
)
parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default=1,
    help="Save a checkpoint every N epochs",
)


"""""" """""" """""" """""" """""" """""" """""" """
Main
""" """""" """""" """""" """""" """""" """""" """"""

def main(args):

    # TODO: Mean flow subtraction
    # TODO: Create dataloader
    
    # Initialise CNNs for spatial and temporal streams
    spatial_model = spatial.CNN(
        height=224, width=224, channels=10, dropout=args.spatial_dropout
    )

    temporal_model = temporal.CNN(
        height=224, width=224, channels=10, dropout=args.temporal_dropout
    )

    # Define loss criterion as softmax cross entropy
    spatial_criterion = nn.CrossEntropyLoss()
    temporal_criterion = nn.CrossEntropyLoss()

    # Define optimisers
    spatial_optimiser = optim.SGD(spatial_model.parameters(), lr=0.001, momentum=0.9)
    temporal_optimiser = optim.SGD(temporal_model.parameters(), lr=0.001, momentum=0.9)

    # Schedulers for decaying the learning rate over time. May be needed later on.
    # spatial_scheduler = lr_scheduler.StepLR(spat_optimizer, step_size=10, gamma=0.1)
    # temporal_scheduler = lr_scheduler.StepLR(temp_optimizer, step_size=10, gamma=0.1)

    # Initialise log writing
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    # Initialise trainer with both CNNs
    trainer = trainer.Trainer(
        spatial_model,
        temporal_model,
        train_loader,
        test_loader,
        spatial_criterion,
        temporal_criterion,
        spatial_optimiser,
        temporal_optimiser,
        summary_writer,
        DEVICE,
        args.checkpoint_frequency,
        args.checkpoint_path,
        args.log_dir,
    )

    # Begin training
    trainer.train(
        args.epochs,
        args.val_frequency,
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

