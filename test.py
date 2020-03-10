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
import models.spatial as spatial
import models.temporal as temporal
import controllers.evaluator as evaluator
import utils.metrics as metrics
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


def main(cfg):

    classes = open(cfg.paths.classes).read().strip().split()

    start_time = datetime.datetime.now()

    # Initialise spatial/temporal CNNs
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

    # Load checkpoints
    spatial_model.load_checkpoint(cfg.name, cfg.paths.checkpoints, cfg.best)
    temporal_model.load_checkpoint(cfg.name, cfg.paths.checkpoints, cfg.best)

    # Initialise test dataloader
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

    # Initialise evaluator
    network_evaluator = evaluator.Evaluator(
        spatial=spatial_model,
        temporal=temporal_model,
        data_loader=test_loader,
        device=DEVICE,
        name=cfg.name,
    )

    print("==> Making predictions")
    predictions_dict = network_evaluator.predict()

    # Create directory for this model's predictions
    if cfg.best:
        cfg.name = f"{cfg.name}_best"
    model_output_path = f"{cfg.paths.output}/{cfg.name}"
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    print("==> Copying frames from dataset")
    copy_frames_to_output(model_output_path, cfg.paths.frames, predictions_dict.keys())

    print("==> Drawing bounding boxes")
    draw_bounding_boxes(
        model_output_path,
        cfg.paths.annotations,
        cfg.dataset.temporal_stack,
        predictions_dict,
        classes,
    )

    print("==> Stitching frames to create final output videos")
    stitch_videos(model_output_path, cfg.paths.frames, predictions_dict)

    # Delete frame directories
    for video in predictions_dict.keys():
        directory_path = f"{model_output_path}/{video}"
        shutil.rmtree(directory_path)

    print("==> Generating confusion matrix")
    metrics.compute_confusion_matrix(predictions_dict, classes, model_output_path)

    print("==> Creating zip file")
    zip_videos(model_output_path, cfg.name)

    print("==> Uploading to AWS S3")
    response = upload_videos(model_output_path, cfg.name, cfg.bucket)

    if response:
        print(f"Output download link: {response}")

    total_time = datetime.datetime.now() - start_time
    print(f"Total time: {total_time.total_seconds()}")


if __name__ == "__main__":
    cfg = ConfigParser().config
    main(cfg)
