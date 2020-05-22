import sys
import os.path
from xml.dom import minidom
import xml.etree.ElementTree as ET

if len(sys.argv) != 7:
    print("error: Arguments should be in the form:\n")
    print(
        "python behaviour_labeller.py [filename] [start frame] [end frame] [ape no 1] [ape no 2] [merge mode]"
    )
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_index_0 = int(sys.argv[4])
ape_index_1 = int(sys.argv[5])
mode = sys.argv[6]

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

    apes_to_merge = []

    if len(root.findall("object")) == 1:
        print(f"Ape {ape_index_1} not found in frame {i}")
        continue

    ape_0 = root.findall("object")[ape_index_0]
    ape_1 = root.findall("object")[ape_index_1]
    ape_box_0 = ape_0[2]
    ape_box_1 = ape_1[2]

    if mode == "overlap":
        root.findall("object")[ape_index_0][2][0].text = str(
            round((float(ape_box_0[0].text) + float(ape_box_1[0].text)) / 2, 2)
        )
        root.findall("object")[ape_index_0][2][1].text = str(
            round((float(ape_box_0[1].text) + float(ape_box_1[1].text)) / 2, 2)
        )
        root.findall("object")[ape_index_0][2][2].text = str(
            round((float(ape_box_0[2].text) + float(ape_box_1[2].text)) / 2, 2)
        )
        root.findall("object")[ape_index_0][2][3].text = str(
            round((float(ape_box_0[3].text) + float(ape_box_1[3].text)) / 2, 2)
        )
    elif mode == "extend":
        if float(ape_box_1[0].text) < float(ape_box_0[0].text):
            root.findall("object")[ape_index_0][2][0].text = ape_box_1[0].text

        if float(ape_box_1[1].text) < float(ape_box_0[1].text):
            root.findall("object")[ape_index_0][2][1].text = ape_box_1[1].text

        if float(ape_box_1[2].text) > float(ape_box_0[2].text):
            root.findall("object")[ape_index_0][2][2].text = ape_box_1[2].text

        if float(ape_box_1[3].text) > float(ape_box_0[3].text):
            root.findall("object")[ape_index_0][2][3].text = ape_box_1[3].text

    # root.findall('object')[ape_index_0].find('activity').text = activity

    root.remove(ape_1)

    tree.write(xml_file_path)
