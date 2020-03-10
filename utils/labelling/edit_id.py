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
old_ape_id = int(sys.argv[4])
new_ape_id = int(sys.argv[5])

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    frame_file_path = f"../dataset/behaviour_annotations/{filename}/{filename}_frame_{i}.xml"
    in_file = open(frame_file_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    for j, ape in enumerate(root.findall("object")):
        if len(ape.findall("id")) == 0:
            print(f"No id label found for ape no {j} for frame {i}")
            continue
        if int(ape.findall("id")[0].text) == old_ape_id:
            ape.findall("id")[0].text = str(new_ape_id)

    tree.write(frame_file_path)

print(
    f"Successfully edited id of ape {old_ape_id} to ape {new_ape_id} from frames {start_frame} to {end_frame}!"
)
