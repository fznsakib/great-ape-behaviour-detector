import sys
import os.path
from xml.dom import minidom

if len(sys.argv) != 7:
    print("error: Arguments should be in the form:\n")
    print(
        "python behaviour_labeller.py [filename] [start frame] [end frame] [ape no] [ape activity]"
    )
    exit()

filename = sys.argv[1]
ape_index = int(sys.argv[2])
ape_activity = sys.argv[3]
start_frame = int(sys.arg[4])
end_frame = int(sys.argv[5])

file_path = f"../dataset/videos/{filename}.mp4"

if not os.path.exists(file_path):
    print(f"error: video {filename} does not exist")
    exit()

# Allowed classes
classes = [
    "standing",
    "sitting",
    "sitting_on_back",
    "walking",
    "running",
    "climbing_up",
    "climbing_down",
    "hanging",
    "camera_interaction",
    "scavenging",
    "eating",
]

if ape_activity not in classes:
    print(f"error: not a valid class")
    exit()

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    xml_file_path = f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{str(i)}.xml"

    print(f"Modifying {xml_file_path}")

    if not os.path.exists(xml_file_path):
        print(f"error: XML file for frame {i} for video {filename} does not exist")
        exit()

    xmldoc = minidom.parse(xml_file_path)
    apes = xmldoc.getElementsByTagName("object")

    if len(apes) == 0:
        print(f"error: no apes found in frame {str(i)}")
        continue

    if len(apes) - 1 < ape_index:
        print(f"error: ape index ({ape_index}) higher than number of apes detected in frame {i}")
        continue

    # Create activity element
    activity_element = xmldoc.createElement("activity")
    activity_text = xmldoc.createTextNode(ape_activity)
    activity_element.appendChild(activity_text)

    ape_start_index = 3

    # Get ape element
    elements = xmldoc.childNodes[0]

    ape = elements.childNodes[ape_start_index + ape_index]

    # Check if activity already exists for this object
    if len(ape.getElementsByTagName("activity")) > 0:
        print(f"error: activity already exists for ape {ape_index}")
        continue

    # Append activity element into ape element
    ape.appendChild(activity_element)

    # Write to file
    file_handle = open(xml_file_path, "w")
    xmldoc.writexml(file_handle)
    file_handle.close()

print(f'Successfully added behaviour "{ape_activity}" for frames {start_frame} to {end_frame}!')
