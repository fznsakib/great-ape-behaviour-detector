import numpy
import glob
import sys
import os
import xml.etree.ElementTree as ET

video = sys.argv[1]

for frame_no in range(1, 361):
    in_file_path = f'../dataset/behaviour_annotations/{video}/{video}_frame_{frame_no}.xml'
    in_file = open(in_file_path)
    
    if (not os.path.exists(in_file_path)):
        print(f'Frame {frame_no} does not exist for video {video}')
        
    tree = ET.parse(in_file)
    root = tree.getroot()
    
    for i, obj in enumerate(root.iter('object')):
        if obj.find('id') == None:
            print(f'No ID found for ape no {i} in frame {frame_no} in video {video}')
            continue
        if obj.find('activity') == None:
            print(f'No activity found for ape no {i} in frame {frame_no} in video {video}')
            continue