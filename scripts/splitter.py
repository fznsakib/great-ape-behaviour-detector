import cv2
import numpy as np
import os

training_set_names = [line.rstrip('\n') for line in open('../dataset/splits/validationdata.txt')]


try:
    if not os.path.exists('../dataset/frames'):
        os.makedirs('../dataset/frames')
except OSError:
    print ('Error: Creating directory of dataset/frames')

for filename in training_set_names:
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