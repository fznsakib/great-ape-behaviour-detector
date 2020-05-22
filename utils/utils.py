import os
import torch
import numpy as np
import shutil
import glob
import cv2
import boto3
import argparse
import json
from tqdm import tqdm
from typing import NamedTuple
from zipfile import ZipFile
from PIL import ImageFont, ImageDraw, Image
from botocore.exceptions import ClientError

from utils.data import *

# Save model to disk
def save_checkpoint(model, is_best_model, name, save_path):

    checkpoint_path = f"{save_path}/{name}"

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    torch.save(model, f"{checkpoint_path}/model")

    if is_best_model:
        shutil.copyfile(f"{checkpoint_path}/model", f"{checkpoint_path}/model_best")


# Copy frames from dataset to prepare for bounding box placement
def copy_frames_to_output(model_output_path, frames_path, videos):
    for video in tqdm(videos, desc="Copying video frames", leave=False, unit="video"):
        if not os.path.exists(f"{model_output_path}/{video}"):
            os.makedirs(f"{model_output_path}/{video}")

        files = glob.glob(f"{frames_path}/rgb/{video}/*.jpg")
        no_of_files = len(files)

        for file_path in tqdm(files, leave=False, unit="frame"):
            shutil.copy(file_path, f"{model_output_path}/{video}")


# Draw bounding boxes with prediction labels onto frames
def draw_bounding_boxes(model_output_path, annotations_path, temporal_stack, predictions, classes):

    with open("assets/colours.json", "r") as j:
        colours = json.loads(j.read())

    # Draw bounding boxes on frames
    for video in tqdm(predictions.keys(), desc="Drawing bounding boxes", leave=False, unit="video"):
        for annotation in tqdm(predictions[video], leave=False):

            ape_id = annotation["ape_id"]
            label = annotation["label"]
            prediction = annotation["prediction"]
            start_frame = annotation["start_frame"]

            # Draw for n following frames
            for j in range(0, temporal_stack):
                image = 0

                frame_no = start_frame + j

                ape = get_ape_by_id(annotations_path, video, frame_no, ape_id)
                ape_coords = get_ape_coordinates(ape)
                ape_species = get_species(ape)

                # Define bounding box text and colour
                label_text = f"{ape_id}:{classes[prediction]}"
                colour = colours[list(colours.keys())[ape_id]]

                # Show visually that prediction is incorrect
                if label != prediction:
                    colour = colours["red"]
                    label_text = f"{label_text} != {classes[label]}"

                image_file_path = f"{model_output_path}/{video}/{video}_frame_{frame_no}.jpg"
                image = cv2.imread(image_file_path)

                # Get coordinates for bounding box
                top_left = (int(ape_coords[0]), int(ape_coords[1]))
                bottom_right = (int(ape_coords[2]), int(ape_coords[3]))

                # Draw main bounding box
                image = cv2.rectangle(image, top_left, bottom_right, colour, 2)

                # Get size of label text in order to create background rectangle to fit
                font = ImageFont.truetype("assets/SF-Bold.otf", 14)
                (width, height), (offset_x, offset_y) = font.font.getsize(label_text)

                # Draw label rectangle and text
                bottom_right = (top_left[0] + width + 3, top_left[1] + height + 5)
                image = cv2.rectangle(image, top_left, bottom_right, colour, -1)

                # Convert the image to RGB (OpenCV uses BGR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Pass the image to PIL to draw the text
                pil_image = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(pil_image)
                draw.text((top_left[0], top_left[1] + offset_y - 1), label_text, font=font)

                # Get back the image to OpenCV
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # Save image
                cv2.imwrite(image_file_path, image)


# Create video from annotated frames
def stitch_videos(model_output_path, frames_path, predictions):
    for i, video in enumerate(
        tqdm(predictions.keys(), desc="Stitching videos", leave=False, unit="video")
    ):
        img_array = []

        no_of_files = len(glob.glob(f"{frames_path}/rgb/{video}/*.jpg"))

        size = 0
        for i in range(1, no_of_files + 1):
            filename = f"{model_output_path}/{video}/{video}_frame_{i}.jpg"

            if not os.path.exists(filename):
                break

            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(
            f"{model_output_path}/{video}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, size
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


# Compress files for disk space
def zip_videos(model_output_path, name):
    file_paths = []
    file_names = []

    for root, directories, files in os.walk(model_output_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
            file_names.append(os.path.basename(filepath))

    with ZipFile(f"{model_output_path}/{name}.zip", "w") as zip:
        for i, file in enumerate(
            tqdm(file_paths, desc="Zipping videos", leave=False, unit="video")
        ):
            zip.write(file, arcname=file_names[i])


# Upload zip file to AWS S3 bucket specified in config
# Returns a download link in the command line
def upload_videos(model_output_path, name, bucket):

    s3 = boto3.client("s3")

    # Validate AWS credentials
    try:
        response = s3.list_buckets()
        print(f"AWS account credentials found. Uploading output to S3 bucket {bucket}")
    except ClientError as e:
        print("AWS account credentials not found. Skipping upload.")
        return

    # Upload zip file to S3 bucket
    s3.upload_file(f"{model_output_path}/{name}.zip", bucket, f"{name}.zip")

    # Set access settings for obtaining download URL
    object_acl = s3.put_object_acl(ACL="public-read", Bucket=bucket, Key=f"{name}.zip")
    response = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": f"{name}.zip", "ResponseExpires": 3600}
    )

    return response


def read_json(file_name):
    file_name = Path(file_name)
    with file_name.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)
