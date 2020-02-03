import cv2
import numpy as np
import os

training_set_names = [line.rstrip('\n') for line in open('dataset/splits/trainingdata.txt')]


try:
    if not os.path.exists('dataset/frames'):
        os.makedirs('dataset/frames')
except OSError:
    print ('Error: Creating directory of dataset/frames')

for filename in training_set_names:
    print(f'Capturing frames of video {filename}.p4')
    # Playing video from file:
    cap = cv2.VideoCapture(f'dataset/videos/{filename}.mp4')

    currentFrame = 1

    while(currentFrame <= 360):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        name = f'./dataset/frames/{filename}_frame_{currentFrame}.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()