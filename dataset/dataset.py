from __future__ import print_function, division
import os
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage import io, transform
from statistics import mode

from utils.data import *


class GreatApeDataset(torch.utils.data.Dataset):
    """
    Great Ape dataset consisting of frames extracted from jungle trap footage.
    Includes RGB frames for spatiality and optical flow for temporality.
    """

    def __init__(
        self, cfg, mode, video_names, classes, device,
    ):
        """
        Args:
            mode (string): Specifies what split of data this will hold (e.g. train, validation, test)
            video_names (string): Path to txt file with the names of videos to sample from.
            classes (string): Path to txt file containing the classes.
            frame_dir (string): Directory with all the images.
            annotations_dir (string): Directory with all the xml annotations.
        """
        super(GreatApeDataset, self).__init__()

        self.mode = mode
        self.sample_interval = cfg.dataset.sample_interval
        self.temporal_stack = cfg.dataset.sequence_length
        self.activity_duration_threshold = cfg.dataset.activity_duration_threshold
        self.video_names = open(video_names).read().strip().split()
        self.classes = classes
        self.frame_dir = cfg.paths.frames
        self.annotations_dir = cfg.paths.annotations
        self.spatial_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ]
        )
        self.temporal_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        (
            self.spatial_augmentation_transform,
            self.temporal_augmentation_transform,
        ) = self.initialise_augmentations(cfg.augmentation)
        self.augmentation_probability = cfg.augmentation.probability

        self.labels = 0
        self.samples = {}
        self.samples_by_class = {}
        self.device = device

        if self.mode == "train":
            self.initialise_dataset()
        elif self.mode == "test" or self.mode == "validation":
            self.initialise_test_dataset()

        self.initialise_labels()
        self.initialise_samples_by_class()

        DEVICE = torch.device("cuda")
        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.to(self.device)

    def __len__(self):
        # If the length of the dataset has not yet been calculated, then do so and store it
        if not hasattr(self, "size"):
            self.size = 0
            for key in self.samples.keys():
                self.size += len(self.samples[key])

        return self.size

    def __getitem__(self, index):

        # Get required sample
        video, ape_id, start_frame, activity = find_sample(self.samples, index)

        """
        Spatial Data
        """
        spatial_sample = []
        transform_probabilities = []

        for i in range(0, len(self.spatial_augmentation_transform)):
            transform_probabilities.append(random.random())

        for i in range(0, self.temporal_stack):
            path = f"{self.frame_dir}/rgb/{video}/{video}_frame_{start_frame + i}.jpg"
            spatial_image = Image.open(path)

            # Get ape and its coordinates
            ape = get_ape_by_id(self.annotations_dir, video, start_frame + i, ape_id)
            coordinates = get_ape_coordinates(ape)

            # Crop around ape
            spatial_image = spatial_image.crop(
                (coordinates[0], coordinates[1], coordinates[2], coordinates[3])
            )

            # Apply augmentation and pre-processing transforms
            if self.mode == "train":
                spatial_data = self.apply_augmentation_transforms(
                    spatial_image, self.spatial_augmentation_transform, transform_probabilities
                )
            spatial_data = self.spatial_transform(spatial_data)

            spatial_sample.append(spatial_data.squeeze_(0))
            spatial_image.close()

        spatial_sample = torch.stack(spatial_sample, dim=0)

        """
        Temporal Data
        """
        # temporal_data = torch.FloatTensor(2 * self.temporal_stack, 224, 224)

        temporal_sample = []
        transform_probabilities = []

        for i in range(0, len(self.temporal_augmentation_transform)):
            transform_probabilities.append(random.random())

        for i in range(0, self.temporal_stack):
            this_data = []

            x_path = f"{self.frame_dir}/horizontal_flow/{video}/{video}_frame_{start_frame + i}.jpg"
            y_path = f"{self.frame_dir}/vertical_flow/{video}/{video}_frame_{start_frame + i}.jpg"

            image_x = Image.open(x_path)
            image_y = Image.open(y_path)

            # Get ape and its coordinates
            ape = get_ape_by_id(self.annotations_dir, video, start_frame + i, ape_id)
            coordinates = get_ape_coordinates(ape)

            # Crop around ape and apply transforms
            image_x = image_x.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))
            image_y = image_y.crop((coordinates[0], coordinates[1], coordinates[2], coordinates[3]))

            # Apply augmentation and pre-processing transforms for entire stack
            if self.mode == "train":
                final_image_x = self.apply_augmentation_transforms(
                    image_x, self.temporal_augmentation_transform, transform_probabilities
                )
                final_image_y = self.apply_augmentation_transforms(
                    image_y, self.temporal_augmentation_transform, transform_probabilities
                )

            final_image_x = self.temporal_transform(final_image_x)
            final_image_y = self.temporal_transform(final_image_y)

            this_data.append(final_image_x.squeeze_(0))
            this_data.append(final_image_y.squeeze_(0))
            stacked_data = torch.stack(this_data, dim=0)

            temporal_sample.append(stacked_data.squeeze(0))

            image_x.close()
            image_y.close()

        temporal_sample = torch.stack(temporal_sample, dim=0)

        """
        Label
        """
        label = self.classes.index(activity)

        """
        Other data
        """
        metadata = {"ape_id": ape_id, "start_frame": start_frame, "video": video}

        return spatial_sample, temporal_sample, label, metadata

    def initialise_dataset(self):
        """
        Creates a dictionary which includes all valid spatial + temporal samples from the dataset.
        The dictionary is exported as a json for later use.
        """

        # Go through every video in dataset
        for video in tqdm(self.video_names, desc=f"Initialising {self.mode} dataset", leave=False):
            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)

            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):

                no_of_frames = len(glob.glob(f"{self.annotations_dir}/{video}/*.xml"))
                frame_no = 1

                # Traverse through every frame to find valid samples
                while frame_no <= no_of_frames:
                    if (no_of_frames - frame_no) < self.activity_duration_threshold:
                        break

                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)

                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        # Check if this ape has the same activity for atleast the next activity_duration_threshold frames
                        current_activity = ape.find("activity").text
                        valid_frames = 1

                        for look_ahead_frame_no in range(frame_no + 1, no_of_frames + 1):
                            ape = get_ape_by_id(
                                self.annotations_dir, video, look_ahead_frame_no, current_ape_id
                            )

                            if (ape) and (ape.find("activity").text == current_activity):
                                valid_frames += 1
                            else:
                                break

                        # If less frames than activity duration threshold, carry on with search
                        if valid_frames < self.activity_duration_threshold:
                            frame_no += valid_frames
                            continue

                        # If this sample meets the required number of frames, break it down into smaller samples with the given interval
                        last_valid_frame = frame_no + valid_frames
                        for valid_frame_no in range(
                            frame_no, last_valid_frame, self.sample_interval
                        ):

                            # For the last valid sample, ensure that there are enough temporal frames with the ape following it
                            if (valid_frame_no + self.sample_interval) >= last_valid_frame:
                                correct_activity = False
                                for temporal_frame in range(valid_frame_no, self.temporal_stack):
                                    ape = get_ape_by_id(
                                        self.annotations_dir, video, temporal_frame, current_ape_id
                                    )
                                    ape_activity = get_activity(ape)
                                    if (
                                        (not ape)
                                        or (ape_activity != current_activity)
                                        or (temporal_frame > no_of_frames)
                                    ):
                                        correct_activity = False
                                        break
                                if correct_activity == False:
                                    break

                            # Check if there are enough frames left
                            if (no_of_frames - valid_frame_no) >= self.temporal_stack:

                                # Insert sample
                                if video not in self.samples.keys():
                                    self.samples[video] = []

                                # Keep count of number of labels
                                self.labels += 1

                                self.samples[video].append(
                                    {
                                        "ape_id": current_ape_id,
                                        "activity": current_activity,
                                        "start_frame": valid_frame_no,
                                    }
                                )

                        frame_no = last_valid_frame

    def initialise_test_dataset(self):
        """
        Creates a dictionary which includes all spatial + temporal samples from the dataset.
        These samples would then be used for a network to classify behaviour of.
        The dictionary is exported as a json for later use.
        """

        # Go through every video in dataset
        for video in tqdm(self.video_names, desc=f"Initialising {self.mode} dataset", leave=False):
            # for video in tqdm(video_names, desc=f'Initialising {self.mode} dataset'):
            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)
            # print(f'video: {video}, no_of_apes: {no_of_apes}')

            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):
                no_of_frames = len(glob.glob(f"{self.annotations_dir}/{video}/*.xml"))
                frame_no = 1

                # Traverse through every frame to get samples
                while frame_no <= no_of_frames:
                    if (no_of_frames - frame_no) < (self.temporal_stack - 1):
                        break

                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)

                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        activities = []
                        insufficient_apes = False

                        # Check that this ape exists for the next n frames
                        for look_ahead_frame_no in range(frame_no, frame_no + self.temporal_stack):
                            ape = get_ape_by_id(
                                self.annotations_dir, video, look_ahead_frame_no, current_ape_id
                            )

                            if ape:
                                activities.append(ape.find("activity").text)
                            else:
                                insufficient_apes = True
                                break

                        # If the ape is not present for enough consecutive frames, then move on
                        if insufficient_apes:
                            # frame_no = look_ahead_frame_no
                            frame_no += self.temporal_stack
                            continue

                        # Get majority activity
                        try:
                            activity = mode(activities)
                        except:
                            activity = activities[0]

                        # Check if there are enough frames left
                        if (no_of_frames - frame_no) >= self.temporal_stack:

                            # Insert sample
                            if video not in self.samples.keys():
                                self.samples[video] = []

                            # Keep count of number of labels
                            self.labels += 1

                            self.samples[video].append(
                                {
                                    "ape_id": current_ape_id,
                                    "activity": activity,
                                    "start_frame": frame_no,
                                }
                            )

                        frame_no += self.temporal_stack

        return

    def initialise_samples_by_class(self):
        """
        Creates a dictionary which includes all spatial + temporal samples from the dataset
        with the classes as keys.
        """
        for class_name in self.classes:
            self.samples_by_class[class_name] = []

        for video in self.samples.keys():
            for annotation in self.samples[video]:
                new_annotation = annotation
                new_annotation["video"] = video
                self.samples_by_class[annotation["activity"]].append(new_annotation)

    def get_no_of_samples_by_class(self):
        """
        Returns a list of number of samples for each class
        """
        samples_by_class = np.unique(self.labels.cpu().numpy(), return_counts=True)[1]

        return samples_by_class

    def initialise_labels(self):
        labels = np.zeros(self.labels, dtype=int)
        i = 0

        for video in self.samples:
            for annotation in self.samples[video]:
                label = self.classes.index(annotation["activity"])
                labels[i] = label
                i += 1

        self.labels = labels

    def _get_label(self, index):
        return self.labels[index]

    def initialise_augmentations(self, augmentation_cfg):

        spatial_aug = []
        temporal_aug = []

        if augmentation_cfg.spatial.colour_jitter:
            spatial_aug.append(
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            )
        if augmentation_cfg.spatial.horizontal_flip:
            spatial_aug.append(transforms.RandomHorizontalFlip(p=1))
        if augmentation_cfg.spatial.rotation:
            spatial_aug.append(transforms.RandomRotation(p=1, degrees=10))

        if augmentation_cfg.temporal.horizontal_flip:
            temporal_aug.append(transforms.RandomHorizontalFlip(p=1))

        return spatial_aug, temporal_aug

    def apply_augmentation_transforms(self, image, augmentations, probabilities):
        for i, aug in enumerate(augmentations):
            if probabilities[i] < self.augmentation_probability:
                image = aug(image)

        return image
