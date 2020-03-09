import numpy
import glob
import sys
import xml.etree.ElementTree as ET
import tqdm
import os

data_list_path = sys.argv[1]
# video_names = [line.rstrip('\n') for line in open(data_list_path)]
# video_names = ['MsU8rQkfps', 'OMZtzH6RwA', 'peFYH4rAIK', 'Rsas1MASXn', 'ys642Q7e7S']
video_names = [data_list_path]

for i, video in enumerate(tqdm.tqdm(video_names, leave=False)):
    
    no_of_files = len(glob.glob(f'../dataset/annotations/{video}/*.xml'))
    
    # print(f'Sorting video ({i}/{len(video_names)}): {video}')

    for frame_no in range(1, no_of_files + 1):
        in_file_path = f'../dataset/behaviour_annotations/{video}/{video}_frame_{frame_no}.xml'
        if (not os.path.isfile(in_file_path)):
            print(f'frame {frame_no} does not exist for video {video}')
            continue
    
        in_file = open(in_file_path)
    
        tree = ET.parse(in_file)
        root = tree.getroot()

        apes = []

        for ape_element in root.findall('object'):
            apes.append(ape_element)

        xmins = []

        for ape_element in apes:
            xmins.append(float(ape_element[2][0].text))

        sorted_ape_indices = list(numpy.argsort(xmins))

        for ape_element in root.findall('object'):
            root.remove(ape_element)

        for index in sorted_ape_indices:
            root.append(apes[index])

        # xmlstr = ET.tostring(tree, encoding="utf-8", method="xml")
        # print(xmlstr.decode("utf-8"))
        
        tree.write(f'../dataset/behaviour_annotations/{video}/{video}_frame_{frame_no}.xml')
