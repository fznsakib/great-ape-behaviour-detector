import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

print("Video to frame converter")
print("Data directory provided should contain a 'videos' directory containing the videos.")

dataset_path = input("Provide path to data directory: ")
data_list_path = input("Provide path to txt file with names of videos to convert to RGB frames: ")

# Get list of video names which need to be split
filenames = open(data_list_path).read().strip().split()

try:
    if not os.path.exists(f"{dataset_path}/rgb"):
        os.makedirs(f"{dataset_path}/rgb")
except OSError:
    print("Error: Creating directory of dataset/rgb")

for filename in tqdm(filenames):

    # Playing video from file:
    cap = cv2.VideoCapture(f"{dataset_path}/videos/{filename}.mp4")
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    currentFrame = 1

    while currentFrame < no_of_frames:

        # Capture frame-by-frame
        ret, frame = cap.read()

        try:
            if not os.path.exists(f"{dataset_path}/rgb/{filename}"):
                os.makedirs(f"{dataset_path}/rgb/{filename}")
        except OSError:
            print(f"Error: Creating directory of dataset/rgb/{filename}")

        # Saves image of the current frame in jpg file
        name = f"{dataset_path}/rgb/{filename}/{filename}_frame_{currentFrame}.jpg"

        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
