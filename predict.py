"""""" """""" """""" """""" """""" """""" """""" """
Imports
""" """""" """""" """""" """""" """""" """""" """"""
import os
import torch
import torchvision
import argparse
import datetime
import numpy as np
import shutil
import json
import glob
import cv2
import tarfile
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm

"""""" """""" """""" """""" """""" """""" """""" """
Custom Library Imports
""" """""" """""" """""" """""" """""" """""" """"""
import spatial
import temporal
import predictor
from dataloader.dataset import GreatApeDataset
from dataloader.data_utils import *
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


def main(args):

    colours = {
        'blue': [255, 191, 0],
        'pink': [128, 128, 240],
        'green': [87, 139, 46],
        'purple': [211, 0, 148],
        'orange': [0, 140, 255],
        'brown' : [30, 105, 210],
        'gray' : [153, 136, 119],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'red': [0, 0, 255]
    }

    if not args.name:
        print("Please specify model name in order to make predictions")
        exit()
    
    classes = open(args.classes).read().strip().split()

    start_time = datetime.datetime.now()

    # Initialise spatial/temporal CNNs
    spatial_model = spatial.CNN(lr=0.001, num_classes=len(classes), channels=3, device=DEVICE)
    temporal_model = temporal.CNN(
        lr=0.001, num_classes=len(classes), channels=args.optical_flow * 2, device=DEVICE,
    )

    # Load checkpoints
    spatial_model.load_checkpoint(args.name, args.checkpoint_path, args.best)
    temporal_model.load_checkpoint(args.name, args.checkpoint_path, args.best)

    # Initialise test dataloader
    test_dataset = GreatApeDataset(
        mode="test",
        sample_interval=5,
        no_of_optical_flow=args.optical_flow,
        activity_duration_threshold=72,
        video_names=f"{args.dataset_path}/splits/validationdata.txt",
        classes=classes,
        frame_dir=f"{args.dataset_path}/frames",
        annotations_dir=f"{args.dataset_path}/annotations",
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
        test_dataset, batch_size=1, shuffle=False, num_workers=args.worker_count,
    )

    # Initialise Predictor
    network_predictor = predictor.Predictor(
        spatial=spatial_model,
        temporal=temporal_model,
        data_loader=test_loader,
        device=DEVICE,
        name=args.name,
    )

    
    print("==> Making predictions")
    predictions_dict = network_predictor.predict()

    # # Create directory for this model's predictions
    model_output_path = f'{args.output_path}/{args.name}'
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # print("==> Copying frames from dataset")
    copy_frames_to_output(model_output_path, args.dataset_path, predictions_dict.keys())
    
    # print("==> Drawing bounding boxes")
    draw_bounding_boxes(args, predictions_dict, classes, colours)
    
    # print("==> Stitching frames to create final output videos")
    stitch_videos(model_output_path, args.dataset_path, predictions_dict)

    # # Delete frame directories
    for video in predictions_dict.keys():
        directory_path = f'{args.output_path}/{args.name}/{video}'
        shutil.rmtree(directory_path)


    print("==> Generating confusion matrix")
    labels = []
    predictions = []
    for video in predictions_dict.keys():
        for annotation in predictions_dict[video]:
            labels.append(annotation['label'])
            predictions.append(annotation['prediction'])

    compute_confusion_matrix(labels, predictions, classes, model_output_path)

    total_time = datetime.datetime.now() - start_time
    print(f'Total time: {total_time.total_seconds()}')


if __name__ == "__main__":
    
    """""" """""" """""" """""" """""" """""" """""" """
    Argument Parser
    """ """""" """""" """""" """""" """""" """""" """"""

    default_dataset_dir = Path(f"{os.getcwd()}/../scratch/dataset")
    default_output_dir = Path(f"{os.getcwd()}/../scratch/output")
    default_classes_dir = Path(f"{default_dataset_dir}/classes.txt")
    default_checkpoints_dir = Path(f"{os.getcwd()}/../scratch/checkpoints")

    parser = argparse.ArgumentParser(
        description="Perform behaviour recognition on great ape videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model to use for prediction
    parser.add_argument(
        "--name", default="", type=str, help="Toggle model checkpointing by providing name"
    )

    parser.add_argument(
        "--best", action="store_true", help="Load the checkpoint with the best accuracy for the model"
    )

    parser.add_argument(
        "--optical-flow",
        default=5,
        type=int,
        help="Number of frames of optical flow to provide the temporal stream",
    )

    # Paths
    parser.add_argument(
        "--dataset-path", default=default_dataset_dir, type=Path, help="Path to root of dataset"
    )
    parser.add_argument(
        "--checkpoint-path",
        default=default_checkpoints_dir,
        type=Path,
        help="Path to root of saved model checkpoints",
    )
    parser.add_argument("--classes", default=default_classes_dir, type=Path, help="Path to classes.txt")
    parser.add_argument(
        "--output-path",
        default=default_output_dir,
        type=Path,
        help="Path to output videos with prediction bounding boxes",
    )

    # Miscellaneous
    parser.add_argument(
        "-j",
        "--worker-count",
        default=cpu_count(),
        type=int,
        help="Number of worker processes used to load data.",
    )

    """""" """""" """""" """""" """""" """""" """""" """
    Call main()
    """ """""" """""" """""" """""" """""" """""" """"""
    main(parser.parse_args())
