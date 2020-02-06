import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# Include classes of dataset
classes = ['chimpanzee', 'gorilla']

# Normalise coordinates to [0, 1] according to dimensions
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

# Convert XML file to label .txt
def convert_annotation(frame_no, image_id):
    
    in_file_path = f'../dataset/annotations/{image_id}/{image_id}_frame_{frame_no}.xml'
    in_file = open(in_file_path)
    
    out_file_path = f'../dataset/labels/{image_id}'
    
    # Create folder for video if it does not exit
    if (not os.path.exists(out_file_path)):
        os.mkdir(out_file_path)
        # exit()
    
    out_file_path = f'{out_file_path}/{image_id}_frame_{frame_no}.txt'
    out_file = open(out_file_path, 'w')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


image_ids_train = open('../dataset/splits/trainingdata.txt').read().strip().split()
image_ids_val = open('../dataset/splits/validationdata.txt').read().strip().split()

list_file_train = open('object_train.txt', 'w')
list_file_val = open('object_val.txt', 'w')

# Convert training annotations
for image_id in image_ids_train:
    print(f'Converting training annotations of {image_id}.mp4')
    no_of_frames = len(os.listdir(f'../dataset/annotations/{image_id}'))

    # do it for all frames
    for frame_no in range(1, no_of_frames + 1):
        convert_annotation(frame_no, image_id)
        list_file_train.write(f'dataset/frames/{image_id}/{image_id}_frame_{frame_no}.jpg\n')
    
list_file_train.close()

# Convert validation annotations
for image_id in image_ids_val:
    print(f'Converting validation annotations of {image_id}.mp4')
    no_of_frames = len(os.listdir(f'../dataset/annotations/{image_id}'))

    for frame_no in range(1, no_of_frames + 1):
        # change the pth
        convert_annotation(frame_no, image_id)
        list_file_val.write(f'dataset/frames/{image_id}/{image_id}_frame_{frame_no}.jpg\n')
    
list_file_val.close()
