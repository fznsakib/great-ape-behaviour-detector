import sys
import os.path
import cv2
import numpy as np
import math
import glob
from tqdm import tqdm
from xml.dom import minidom


# Draw bounding boxes on individual frames
def get_frames_with_boxes(videos):
    for i, video in enumerate(tqdm(videos)):
        frame_file_path = f"../dataset/frames/{video}"
        # print(f'Drawing boxes for video no {i}: {video}.mp4')

        for i in range(1, 361):
            xml_file_path = f"../dataset/behaviour_annotations/{video}/{video}_frame_{str(i)}.xml"

            if not os.path.exists(xml_file_path):
                print(f"error: video {video} frame no {i} does not exist")
                continue

            xmldoc = minidom.parse(xml_file_path)
            boxes = xmldoc.getElementsByTagName("bndbox")

            image_file_path = f"{frame_file_path}/{video}_frame_{i}.jpg"
            image = cv2.imread(image_file_path)

            # Get coordinates
            for box in boxes:
                xmin = math.floor(float(box.childNodes[0].childNodes[0].nodeValue))
                ymin = math.floor(float(box.childNodes[1].childNodes[0].nodeValue))
                xmax = math.floor(float(box.childNodes[2].childNodes[0].nodeValue))
                ymax = math.floor(float(box.childNodes[3].childNodes[0].nodeValue))

                image = cv2.rectangle(
                    image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2
                )

            # Write frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2

            # Using cv2.putText() method
            image = cv2.putText(image, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Save image
            cv2.imwrite(image_file_path, image)


# Get all frames with bounding box and stitch into video
def stitch_to_video(videos):
    for i, video in enumerate(tqdm(videos)):
        img_array = []

        files = sorted(glob.glob(f"../dataset/frames/{video}/*.jpg"))
        no_of_files = len(files)

        # print(f'Stitching frames for video no {i}: {video}.mp4')

        for i in range(1, no_of_files + 1):
            filename = f"../dataset/frames/{video}/{video}_frame_{i}.jpg"
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(
            f"../dataset/framed_videos/{video}.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 24, size
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("error: Please provide the video that needs to be drawn on.")

    # videos = os.listdir('../dataset/frames')
    videos = [sys.argv[1]]
    # videos = open(sys.argv[1]).read().strip().split()
    get_frames_with_boxes(videos)
    stitch_to_video(videos)
