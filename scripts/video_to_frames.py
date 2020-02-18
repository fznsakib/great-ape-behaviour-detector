import cv2
import numpy as np
import os
import sys

if (len(sys.argv) != 2):
    print('error: Arguments should be in the form:\n')
    print('python splitter.py [path to data list]')
    
data_list_path = sys.argv[1]

# Get list of video names which need to be split
filenames = [line.rstrip('\n') for line in open(data_list_path)]

try:
    if not os.path.exists('../dataset/frames'):
        os.makedirs('../dataset/frames')
except OSError:
    print ('Error: Creating directory of dataset/frames')

for filename in filenames:
    print(f'Capturing frames of video {filename}.mp4')
    
    # Playing video from file:
    cap = cv2.VideoCapture(f'../dataset/videos/{filename}.mp4')
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    currentFrame = 1

    while(currentFrame <= no_of_frames):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        try:
            if not os.path.exists(f'../dataset/frames/{filename}'):
                os.makedirs(f'../dataset/frames/{filename}')
        except OSError:
            print ('Error: Creating directory of dataset/frames')

        # Saves image of the current frame in jpg file
        name = f'../dataset/frames/{filename}/{filename}_frame_{currentFrame}.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()