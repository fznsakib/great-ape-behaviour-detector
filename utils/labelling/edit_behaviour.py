import sys
import os.path
import xml.etree.ElementTree as ET


if len(sys.argv) != 6:
    print("error: Arguments should be in the form:\n")
    print(
        "python behaviour_labeller.py [filename] [start frame] [end frame] [ape id] [ape activity]"
    )
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_id = int(sys.argv[4])
ape_activity = sys.argv[5]

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    frame_file_path = f"../../../scratch/data/annotations/{filename}/{filename}_frame_{i}.xml"
    in_file = open(frame_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for ape in root.findall("object"):
        if int(ape.findall("id")[0].text) == ape_id:
            if len(ape.findall("activity")) == 0:
                print(f"No activity label found for ape {ape_id} for frame {i}")
            ape.findall("activity")[0].text = ape_activity

    tree.write(frame_file_path)

print(f"Successfully edited activity of ape {ape_id} from frames {start_frame} to {end_frame}!")
