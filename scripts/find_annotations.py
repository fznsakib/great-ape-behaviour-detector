import sys
import os.path
from xml.dom import minidom
import glob

data_list_path = sys.argv[1]
activity = sys.argv[2]

filenames = [line.rstrip('\n') for line in open(data_list_path)]
annotations = {}

for filename in filenames:
    print(f'going through video {filename}')
    file_path = f'../dataset/behaviour_annotations/{filename}'
    
    no_of_files = len(glob.glob(f'{file_path}/*.xml'))
    
    for i in range(1, no_of_files + 1):
        xml_file_path = f'{file_path}/{filename}_frame_{str(i)}.xml'
        
        xmldoc = minidom.parse(xml_file_path)
        apes = xmldoc.getElementsByTagName('activity')
                
        for ape in apes:
            ape_activity = ape.firstChild.nodeValue
            # print(ape.firstChild.nodeValue)
            
            if ape_activity == activity:
                if filename not in annotations:
                    annotations[filename] = [i]
                else:
                    annotations[filename].append(i)

print(annotations)
exit()
