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
import models.network as network
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
    cnn = network.CNN(cfg=cfg, num_classes=len(classes), device=DEVICE)

    # Load checkpoints
    cnn.load_checkpoint(cfg.name, cfg.paths.checkpoints, cfg.best)

    # Initialise test dataloader
    test_dataset = GreatApeDataset(
        cfg=cfg,
        mode="test",
        video_names=f"{cfg.paths.splits}/testdata.txt",
        classes=classes,
        device=DEVICE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.worker_count,
    )

    # Initialise evaluator
    network_evaluator = evaluator.Evaluator(
        cnn=cnn, data_loader=test_loader, device=DEVICE, name=cfg.name,
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
        cfg.dataset.sequence_length,
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
