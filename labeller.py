import sys
import os.path
from xml.dom import minidom

if (len(sys.argv) != 6):
    print('error: Arguments should be in the form:\n[filename] [start frame] [end frame] [ape no] [ape activity]')
    exit()

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])
ape_index = int(sys.argv[4])
ape_activity = sys.argv[5]

file_path = f'dataset/videos/{filename}.mp4'

if (not os.path.exists(file_path)):
    print(f'error: video {filename} does not exist')
    exit()

# Go through xml files for every frame in range
for i in range(start_frame, end_frame + 1):
    xml_file_path = f'dataset/labels/{filename}_frame_{str(i)}.xml'

    if (not os.path.exists(xml_file_path)):
        print(f'error: XML file for frame {i} for video {filename} does not exist')
        exit()

    xmldoc = minidom.parse(xml_file_path)
    apes = xmldoc.getElementsByTagName('object')

    if (len(apes) == 0):
        print(f'error: no apes found in frame {str(i)}')
        exit()
    
    if (len(apes) - 1 < ape_index):
        print(f'error: ape index ({ape_index}) higher than number of apes detected in frame {i}')
        exit()

    # Create activity element
    activity_element = xmldoc.createElement("activity")
    activity_text = xmldoc.createTextNode(ape_activity)
    activity_element.appendChild(activity_text)

    ape_start_index = 4

    # Get ape element
    elements = xmldoc.childNodes[0]
    ape = elements.childNodes[ape_start_index + ape_index]

    # Check if activity already exists for this object
    if (len(ape.getElementsByTagName('activity')) > 0):
        print(f'error: activity already exists for ape {ape_index}')
        exit()
    
    # Append activity element into ape element
    ape.appendChild(activity_element)

    # Write to file
    file_handle = open(xml_file_path,"w")
    xmldoc.writexml(file_handle)
    file_handle.close()





