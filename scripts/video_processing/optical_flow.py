import cv2
import numpy as np
import tqdm
import os
import sys

frames_path = f'../mini_dataset/frames'
data_list_path = sys.argv[1]
video_names = open(data_list_path).read().strip().split()

for video in tqdm.tqdm(video_names):
    for frame in tqdm.tqdm(range(1, 361)):
        frame_1_path = f'{frames_path}/rgb/{video}/{video}_frame_{frame}.jpg'
        frame_2_path = f'{frames_path}/rgb/{video}/{video}_frame_{frame + 1}.jpg'
        
        if not os.path.exists(frame_2_path):
            print(f'Reached the final frame available of video {video}!')
            continue
            
        if not os.path.exists(frames_path):
            print(f'Frame {frame} does not exist for video {video}')
            continue
        
        frame_1 = cv2.imread(frame_1_path)
        frame_2 = cv2.imread(frame_2_path)
        
        prev_frame = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        
        horizontal_frame_path = f'{frames_path}/horizontal_flow/{video}'
        if not os.path.exists(horizontal_frame_path):
            os.mkdir(horizontal_frame_path)
        
        vertical_frame_path = f'{frames_path}/vertical_flow/{video}'
        if not os.path.exists(vertical_frame_path):
            os.mkdir(vertical_frame_path)

        cv2.imwrite(f'{horizontal_frame_path}/{video}_frame_{frame}.jpg', horz)
        cv2.imwrite(f'{vertical_frame_path}/{video}_frame_{frame}.jpg', vert)