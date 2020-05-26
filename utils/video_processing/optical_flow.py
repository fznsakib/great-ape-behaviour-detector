import cv2
import numpy as np
import tqdm
import os
import sys
import shutil

print("RGB to TV-L1 optical flow converter")

frames_path = input("Provide path to frames directory: ")
data_list_path = input("Provide path to txt file with names of videos to convert to optical flow: ")

video_names = open(data_list_path).read().strip().split()

# Create directories to save optical flow images to
if not os.path.exists(f"{frames_path}/horizontal_flow"):
    os.mkdir(f"{frames_path}/horizontal_flow")

if not os.path.exists(f"{frames_path}/vertical_flow"):
    os.mkdir(f"{frames_path}/vertical_flow")

# Compute optical flow for every frame of every video
for video in tqdm.tqdm(video_names, leave=False):
    for frame in tqdm.tqdm(range(1, 361), leave=False):
        frame_1_path = f"{frames_path}/rgb/{video}/{video}_frame_{frame}.jpg"
        frame_2_path = f"{frames_path}/rgb/{video}/{video}_frame_{frame + 1}.jpg"

        # Check if optical flow frame already exists
        optical_flow_frame_path = f"{frames_path}/horizontal_flow/{video}/{video}_frame_{frame}.jpg"
        if os.path.exists(optical_flow_frame_path):
            continue

        if not os.path.exists(frame_2_path):
            continue

        if not os.path.exists(frames_path):
            continue

        # Get pair of consecutive frames to extract optical flow from
        frame_1 = cv2.imread(frame_1_path)
        frame_2 = cv2.imread(frame_2_path)

        prev_frame = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        # Compute and normalise TV-L1 optical flow
        dtvl1 = cv2.optflow.createOptFlow_DualTVL1()
        flow = dtvl1.calc(prev_frame, next_frame, None)

        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype("uint8")
        vert = vert.astype("uint8")

        horizontal_frame_path = f"{frames_path}/horizontal_flow/{video}"
        if not os.path.exists(horizontal_frame_path):
            os.mkdir(horizontal_frame_path)

        vertical_frame_path = f"{frames_path}/vertical_flow/{video}"
        if not os.path.exists(vertical_frame_path):
            os.mkdir(vertical_frame_path)

        # Save optical flow images
        cv2.imwrite(f"{horizontal_frame_path}/{video}_frame_{frame}.jpg", horz)
        cv2.imwrite(f"{vertical_frame_path}/{video}_frame_{frame}.jpg", vert)

        # Pad final image
        frame_3_path = f"{frames_path}/rgb/{video}/{video}_frame_{frame + 2}.jpg"
        if not os.path.exists(frame_3_path):
            cv2.imwrite(f"{horizontal_frame_path}/{video}_frame_{frame + 1}.jpg", horz)
            cv2.imwrite(f"{vertical_frame_path}/{video}_frame_{frame+ 1}.jpg", vert)
