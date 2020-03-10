import sys
import os.path
from xml.dom import minidom
import xml.etree.ElementTree as ET

if len(sys.argv) != 5:
    print("error: Arguments should be in the form:\n")
    print("python behaviour_labeller.py [filename] [start frame] [end frame] [ape no]")
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_index = int(sys.argv[4])

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    xml_file_path = f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{str(i)}.xml"

    print(f"Modifying {xml_file_path}")

    if not os.path.exists(xml_file_path):
        print(f"error: XML file for frame {i} for video {filename} does not exist")
        continue

    in_file = open(xml_file_path)

    tree = ET.parse(in_file)
    root = tree.getroot()

    for i, ape_element in enumerate(root.findall("object")):
        if ape_index == -1:
            if ape_element.find("id") == None:
                root.remove(ape_element)
                continue
        if ape_index == i:
            root.remove(ape_element)

    tree.write(xml_file_path)
