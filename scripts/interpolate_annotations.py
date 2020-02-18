import shutil
import glob
import sys
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tqdm import trange

data_list_path = sys.argv[1]
video_names = [line.rstrip('\n') for line in open(data_list_path)]

video = '1EQeqW53P0'

no_of_files = len(glob.glob(f'../dataset/annotations/{video}/*.xml'))

last_interpolated_frame = 0
# Loop through all 106 existing frames
for frame_no in trange(1, 106 + 1):
    
    frames_to_interpolate = 3

    if frame_no % 3 == 1:
        frames_to_interpolate = 4
    elif frame_no % 15 == 0:
        frames_to_interpolate = 4
        
    
    output_path = f'../dataset/annotations/{video}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    annotations = []
    
    # Get XML root elements for current and next frames
    for i in range(0, frames_to_interpolate):
        in_file_path = f'../dataset/annotations/{video}/{video}_frame_{frame_no + i}.xml'
        in_file = open(in_file_path)
        annotations.append(ET.parse(in_file).getroot())  
            
    this_frame_apes = []
    next_frame_apes = []
    
    # Get all ape elements for current and next frames
    for ape_element in annotations[0].findall('object'):
        this_frame_apes.append(ape_element)

    for ape_element in annotations[1].findall('object'):
        next_frame_apes.append(ape_element)
    
    this_frame_ape_coords = {}
    next_frame_ape_coords = {}
    
    # Get coordinates for all apes for current and next frames, along with their IDs
    for ape in this_frame_apes:
        coords = []
        ape_id = int(ape[3].text)
        for i in range(0, 4):
            coords.append(float(ape[2][i].text))
        
        this_frame_ape_coords[ape_id] = coords

    for ape in next_frame_apes:
        coords = []
        ape_id = int(ape[3].text)
        for i in range(0, 4):
            coords.append(float(ape[2][i].text))
        
        next_frame_ape_coords[ape_id] = coords
        
    second_frame_ape_coords = {}
    third_frame_ape_coords = {}
    fourth_frame_ape_coords = {}
    
    # Find interpolated coordinates for the subsequent second and third frame of the current frame
    for ape_id in this_frame_ape_coords.keys():
        if ape_id in next_frame_ape_coords.keys():
            for i in range(1, frames_to_interpolate):
                xmin = round(this_frame_ape_coords[ape_id][0] + ((next_frame_ape_coords[ape_id][0] -  this_frame_ape_coords[ape_id][0]) * (i/frames_to_interpolate)), 2)
                ymin = round(this_frame_ape_coords[ape_id][1] + ((next_frame_ape_coords[ape_id][1] -  this_frame_ape_coords[ape_id][1]) * (i/frames_to_interpolate)), 2)
                xmax = round(this_frame_ape_coords[ape_id][2] + ((next_frame_ape_coords[ape_id][2] -  this_frame_ape_coords[ape_id][2]) * (i/frames_to_interpolate)), 2)
                ymax = round(this_frame_ape_coords[ape_id][3] + ((next_frame_ape_coords[ape_id][3] -  this_frame_ape_coords[ape_id][3]) * (i/frames_to_interpolate)), 2)
                
                if i == 1: second_frame_ape_coords[ape_id] = [xmin, ymin, xmax, ymax]
                if i == 2: third_frame_ape_coords[ape_id] = [xmin, ymin, xmax, ymax]
                if i == 3: fourth_frame_ape_coords[ape_id] = [xmin, ymin, xmax, ymax]

    # Copy the current annotation across three times and store output path for each frame annotation
    output_paths = []
    # for i in range((frame_no * 3) - 2, (frame_no * 3) + 1):
    for i in range(last_interpolated_frame + 1, last_interpolated_frame + frames_to_interpolate + 1):
        output_path = f'../dataset/annotations/{video}/{video}_frame_{i}.xml'
        output_paths.append(output_path)
        shutil.copy(f'../dataset/annotations/{video}/{video}_frame_{frame_no}.xml', output_path)
    
    if frame_no == 106:
        exit()
        
    # Update second frame annotation
    tree = ET.parse(output_paths[1])
    root = tree.getroot()
    
    for ape_element in root.findall('object'):
        ape_id = int(ape_element.find('id').text)
        if ape_id in second_frame_ape_coords:
            ape_element.find('bndbox/xmin').text = str(second_frame_ape_coords[ape_id][0])
            ape_element.find('bndbox/ymin').text = str(second_frame_ape_coords[ape_id][1])
            ape_element.find('bndbox/xmax').text = str(second_frame_ape_coords[ape_id][2])
            ape_element.find('bndbox/ymax').text = str(second_frame_ape_coords[ape_id][3])
        else:
            root.remove(ape_element)

    tree.write(output_paths[1])
    
    # Update third frame annotation
    tree = ET.parse(output_paths[2])
    root = tree.getroot()
    
    for ape_element in root.findall('object'):
        ape_id = int(ape_element.find('id').text)
        if ape_id in third_frame_ape_coords:
            ape_element.find('bndbox/xmin').text = str(third_frame_ape_coords[ape_id][0])
            ape_element.find('bndbox/ymin').text = str(third_frame_ape_coords[ape_id][1])
            ape_element.find('bndbox/xmax').text = str(third_frame_ape_coords[ape_id][2])
            ape_element.find('bndbox/ymax').text = str(third_frame_ape_coords[ape_id][3])
        else:
            root.remove(ape_element)

    tree.write(output_paths[2])
    
    # Update fourth frame annotation if required
    if frames_to_interpolate == 4:
        tree = ET.parse(output_paths[3])
        root = tree.getroot()
        
        for ape_element in root.findall('object'):
            ape_id = int(ape_element.find('id').text)
            if ape_id in third_frame_ape_coords:
                ape_element.find('bndbox/xmin').text = str(fourth_frame_ape_coords[ape_id][0])
                ape_element.find('bndbox/ymin').text = str(fourth_frame_ape_coords[ape_id][1])
                ape_element.find('bndbox/xmax').text = str(fourth_frame_ape_coords[ape_id][2])
                ape_element.find('bndbox/ymax').text = str(fourth_frame_ape_coords[ape_id][3])
            else:
                root.remove(ape_element)

        tree.write(output_paths[3])
    
    last_interpolated_frame += frames_to_interpolate