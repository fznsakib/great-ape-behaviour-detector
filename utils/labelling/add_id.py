import sys
import os.path
from xml.dom import minidom
import xml.etree.ElementTree as ET

if len(sys.argv) != 6:
    print("error: Arguments should be in the form:\n")
    print("python behaviour_labeller.py [filename] [start frame] [end frame] [ape no] [ape id]")
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_index = int(sys.argv[4])
ape_id = sys.argv[5]

if not ape_id.isnumeric():
    print("error: ID should be an integer")
    exit()

file_path = f"../dataset/videos/{filename}.mp4"

if not os.path.exists(file_path):
    print(f"error: video {filename} does not exist")
    exit()

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    xml_file_path = f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{str(i)}.xml"

    print(f"Modifying {xml_file_path}")

    if not os.path.exists(xml_file_path):
        print(f"error: XML file for frame {i} for video {filename} does not exist")
        continue

    xmldoc = minidom.parse(xml_file_path)
    apes = xmldoc.getElementsByTagName("object")

    if len(apes) == 0:
        print(f"error: no apes found in frame {str(i)}")
        continue

    if len(apes) - 1 < ape_index:
        print(f"error: ape index ({ape_index}) higher than number of apes detected in frame {i}")
        continue

    # Create activity element
    id_element = xmldoc.createElement("id")
    id_text = xmldoc.createTextNode(ape_id)
    id_element.appendChild(id_text)

    ape_start_index = 3

    # Get ape element
    elements = xmldoc.childNodes[0]

    ape = elements.childNodes[ape_start_index + ape_index]

    # Check if activity already exists for this object
    if len(ape.getElementsByTagName("id")) > 0:
        print(f"error: id already exists for ape {ape_index}")
        continue

    # Append activity element into ape element
    ape.appendChild(id_element)

    # Write to file
    file_handle = open(xml_file_path, "w")
    xmldoc.writexml(file_handle)
    file_handle.close()

print(
    f"Successfully assigned ape no {ape_index} with ID {ape_id} for frames {start_frame} to {end_frame}!"
)
