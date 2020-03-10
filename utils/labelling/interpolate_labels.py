import shutil
import glob
import sys
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from tqdm import trange

if len(sys.argv) != 5:
    print("error: Arguments should be in the form:\n")
    print("python behaviour_labeller.py [filename] [ape id] [start frame] [end frame]")
    exit()

filename = sys.argv[1]
ape_id = int(sys.argv[2])
start_frame = int(sys.argv[3])
end_frame = int(sys.argv[4])

start_frame_file_path = (
    f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{start_frame}.xml"
)
in_file = open(start_frame_file_path)

start_frame_tree = ET.parse(in_file)
start_frame_root = start_frame_tree.getroot()

end_frame_file_path = (
    f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{end_frame}.xml"
)
in_file = open(end_frame_file_path)

end_frame_tree = ET.parse(in_file)
end_frame_root = end_frame_tree.getroot()

# Get first and last frame apes/coordinates
start_frame_ape = 0
end_frame_ape = 0

for ape in start_frame_root.findall("object"):
    if int(ape.findall("id")[0].text) == ape_id:
        start_frame_ape = ape

for ape in end_frame_root.findall("object"):
    if int(ape.findall("id")[0].text) == ape_id:
        end_frame_ape = ape

start_frame_ape_coords = []
end_frame_ape_coords = []

for coord in start_frame_ape.findall("bndbox")[0]:
    start_frame_ape_coords.append(float(coord.text))
for coord in end_frame_ape.findall("bndbox")[0]:
    end_frame_ape_coords.append(float(coord.text))

frame_coordinates = {}

# Interpolate coordinates across frames
for i in range(start_frame + 1, end_frame):
    no_of_frames = end_frame - start_frame
    frame_no = i - start_frame
    coords = []
    xmin = round(
        start_frame_ape_coords[0]
        + ((end_frame_ape_coords[0] - start_frame_ape_coords[0]) * (frame_no / no_of_frames)),
        2,
    )
    ymin = round(
        start_frame_ape_coords[1]
        + ((end_frame_ape_coords[1] - start_frame_ape_coords[1]) * (frame_no / no_of_frames)),
        2,
    )
    xmax = round(
        start_frame_ape_coords[2]
        + ((end_frame_ape_coords[2] - start_frame_ape_coords[2]) * (frame_no / no_of_frames)),
        2,
    )
    ymax = round(
        start_frame_ape_coords[3]
        + ((end_frame_ape_coords[3] - start_frame_ape_coords[3]) * (frame_no / no_of_frames)),
        2,
    )
    coords = [xmin, ymin, xmax, ymax]
    frame_coordinates[i] = coords

# Insert ape annotation with new coordinates
for i in range(start_frame + 1, end_frame):
    coords = frame_coordinates[i]
    frame_file_path = f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{i}.xml"
    in_file = open(frame_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for ape in root.findall("object"):
        if int(ape.findall("id")[0].text) == ape_id:
            print(f"Ape {ape_id} already exists in frame {i}")
            exit()

    start_frame_ape.find("bndbox/xmin").text = str(coords[0])
    start_frame_ape.find("bndbox/ymin").text = str(coords[1])
    start_frame_ape.find("bndbox/xmax").text = str(coords[2])
    start_frame_ape.find("bndbox/ymax").text = str(coords[3])

    root.append(start_frame_ape)
    tree.write(frame_file_path)

print(f"Successfully interpolated ape {ape_id} from frames {start_frame} to {end_frame}!")


exit()
