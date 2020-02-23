from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import xml.etree.ElementTree as ET


class GreatApeDataset(torch.utils.data.Dataset):
    """
    Great Ape dataset consisting of frames extracted from jungle trap footage.
    Includes RGB frames for spatiality and optical flow for temporality.
    """
    
    def __init__(self, video_names, frame_dir, annotations_dir, transform=None):
        """
        Args:
            video_names (string): Path to the txt file with the names of videos to sample from.
            frame_dir (string): Directory with all the images.
            annotations_dir (string): Directory with all the xml annotations.
        """
        # super(GreatApeDataset, self).__init__()
        self.video_names = open(video_names).read().strip().split()
        self.frame_dir = frame_dir
        self.annotations_dir = annotations_dir

    def __len__(self):
        # TODO: Go through every frame and find the total number frames with apes?
        return
    
    def __getitem__(self, index):
        return 

"""

1. 1 RGB images (spatial)
2. 20 Optical flow images (10 vertical, 10 horizontal) (temporal)
3. Get activity
4. Find ape coordinates, and crop frame
5. Resize frame to 224 x 224

"""


"""
len(spatial) = {
    video_name = {
        frame_no = [
            ape_0
            ape_1
        ]
    }
}

len(temporal) = {
    video_name: frame_no : {
        
    }
}

"""