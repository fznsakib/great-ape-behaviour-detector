import sys
import os.path
import xml.etree.ElementTree as ET


if len(sys.argv) != 7:
    print("error: Arguments should be in the form:\n")
    print(
        "python behaviour_labeller.py [filename] [start frame] [end frame] [ape index] [coordinate index] [factor]"
    )
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_index = int(sys.argv[4])
coordinate_index = int(sys.argv[5])
factor = float(sys.argv[6])

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

    ape = root.findall("object")[ape_index][2]

    new_coord = round((float(ape[coordinate_index].text)) * factor, 2)

    height = root.find("size")[0].text
    width = root.find("size")[1].text

    if new_coord < 0:
        new_coord = 0

    if coordinate_index == 2:
        if new_coord > float(width):
            new_coord = float(width)

    if coordinate_index == 3:
        if new_coord > float(height):
            new_coord = float(height)

    # new_coord = 95.00

    root.findall("object")[ape_index][2][coordinate_index].text = str(new_coord)

    tree.write(xml_file_path)


print(
    f"Successfully edited coordinate of ape no {ape_index} from frames {start_frame} to {end_frame}!"
)
