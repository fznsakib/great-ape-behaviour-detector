from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import xml.etree.ElementTree as ET
import tqdm
import json
from utils import *


class GreatApeDataset(torch.utils.data.Dataset):
    """
    Great Ape dataset consisting of frames extracted from jungle trap footage.
    Includes RGB frames for spatiality and optical flow for temporality.
    """
    
    def __init__(self, mode, sample_interval, no_of_optical_flow, activity_duration_threshold, video_names, classes, frame_dir, annotations_dir, transform=None):
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
        self.sample_interval = sample_interval
        self.no_of_optical_flow = no_of_optical_flow
        self.activity_duration_threshold = activity_duration_threshold
        self.video_names = open(video_names).read().strip().split()
        self.classes = open(classes).read().strip().split()
        self.frame_dir = frame_dir
        self.annotations_dir = annotations_dir
        self.samples = {}
        
        self.initialise_dataset()
        
        # if mode == 'train':
        #     self.initialise_training_dataset()
        # elif mode == 'validation':
        #     self.initialise_validation_dataset()

    def __len__(self):
        # If the length of the dataset has not yet been calculated, then do so and store it
        if not hasattr(self, 'size'):
            self.size = 0
            for key in self.samples.keys():
                self.size += len(self.samples[key])
        
        return self.size
    
    def __getitem__(self, index):
        return
    
    def initialise_dataset(self):
        """
        Creates a dictionary which includes all valid spatial + temporal samples from the dataset.
        The dictionary is exported as a json for later use.
        """
        
        # Check if required json for this subset of data already exists
        if os.path.isfile(f'{mode}_samples.json'):
            with open(f'{mode}_samples.json', 'r') as f:
                self.samples = json.load(f)
                return
                
        # Go through every video in dataset
        for video in tqdm.tqdm(self.video_names):
            # Count how many apes are present in the video
            no_of_apes = get_no_of_apes(self.annotations_dir, video)
            
            # Go through each ape by id for possible samples
            for current_ape_id in range(0, no_of_apes + 1):
                
                no_of_frames = len(glob.glob(f'{self.annotations_dir}/{video}/*.xml'))
                frame_no = 1
                
                # Traverse through every frame to find valid samples
                while(frame_no <= no_of_frames):
                    if (no_of_frames - frame_no) < self.activity_duration_threshold:
                        break
                    
                    # Find first instance of ape by id
                    ape = get_ape_by_id(self.annotations_dir, video, frame_no, current_ape_id)
                                        
                    if not ape:
                        frame_no += 1
                        continue
                    else:
                        # Check if this ape has the same activity for atleast the next activity_duration_threshold frames
                        current_activity = ape.find('activity').text
                        valid_frames = 1
                        
                        for look_ahead_frame_no in range(frame_no + 1, no_of_frames + 1):
                            ape = get_ape_by_id(self.annotations_dir, video, look_ahead_frame_no, current_ape_id)

                            if (ape) and (ape.find('activity').text == current_activity):
                                valid_frames += 1
                            else:
                                break
                                                    
                        # If less than 72 frames, carry on with search
                        if (valid_frames < self.activity_duration_threshold):
                            frame_no += valid_frames
                            continue
                        
                        # If this sample meets the required numnber of frames, break it down into smaller samples with the given interval
                        last_valid_frame = frame_no + valid_frames
                        for valid_frame_no in range(frame_no, last_valid_frame, self.sample_interval):
                            # Check if there are enough frames left
                            if (no_of_frames - valid_frame_no) >= self.no_of_optical_flow:
                                
                                # Insert sample
                                if video not in self.samples.keys():
                                    self.samples[video] = []
                                
                                self.samples[video].append({
                                    'id': current_ape_id,
                                    'activity': current_activity,
                                    'start_frame': valid_frame_no
                                })
                            
                        frame_no = last_valid_frame
            
        samples_json = json.dumps(self.samples)
        f = open(f'{mode}_samples.json', 'w')
        f.write(samples_json)
        f.close()

                    
"""

1. 1 RGB images (spatial)
2. 10 Optical flow images (5 vertical, 5 horizontal) (temporal)
3. Get activity
4. Find ape coordinates, and crop frame
5. Resize frame to 224 x 224
6. Normalise as suggested by pytorch vgg net

"""

"""
samples = {
    video_name: {
        id: 1,
        activity: sitting,
        start_frame: 50
    }
}
"""

if __name__ == '__main__':
    
    mode = 'train'
    sample_interval = 10
    no_of_optical_flow = 5
    activity_duration_threshold = 72
    video_names = '../mini_dataset/splits/trainingdata.txt'
    classes = '../mini_dataset/classes.txt'
    frame_dir = '../mini_dataset/frames'
    annotations_dir = '../mini_dataset/annotations'
    
    dataset = GreatApeDataset(mode, sample_interval, no_of_optical_flow, activity_duration_threshold, video_names, classes, frame_dir, annotations_dir)
    print(dataset.__len__())