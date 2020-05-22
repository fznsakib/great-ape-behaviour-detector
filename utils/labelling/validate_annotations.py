import numpy
import glob
import sys
import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

# video = sys.argv[1]
videos = open("../dataset/splits/alldata.txt").read().strip().split()
id_dict = {}
behaviour_dict = {}

for video in tqdm(videos):
    for frame_no in range(1, 361):
        in_file_path = f"../dataset/behaviour_annotations/{video}/{video}_frame_{frame_no}.xml"

        if not os.path.exists(in_file_path):
            continue

        in_file = open(in_file_path)

        tree = ET.parse(in_file)
        root = tree.getroot()

        for i, obj in enumerate(root.iter("object")):
            if obj.find("id") == None:
                if video not in id_dict.keys():
                    id_dict[video] = []

                id_dict[video].append(frame_no)
                continue
            if obj.find("activity") == None:
                if video not in behaviour_dict.keys():
                    behaviour_dict[video] = []

                behaviour_dict[video].append(frame_no)
                continue

id_json = json.dumps(id_dict)
f = open(f"id.json", "w")
f.write(id_json)
f.close()

behaviour_json = json.dumps(behaviour_dict)
f = open(f"behaviour.json", "w")
f.write(behaviour_json)
f.close()

print("---------------------")

print("---------------------")

print(len(id_dict.keys()))
print(len(behaviour_dict.keys()))
