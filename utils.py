import os
import torch
import numpy as np
import shutil
import glob
import cv2
import boto3
import argparse
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import NamedTuple
from zipfile import ZipFile
from sklearn.metrics import confusion_matrix
from PIL import ImageFont, ImageDraw, Image  
from botocore.exceptions import ClientError

from dataloader.data_utils import *
    
# Compute the fusion logits across the spatial and temporal stream to perform average fusion
def average_fusion(spatial_logits, temporal_logits):
    spatial_logits = spatial_logits.cpu().detach().numpy()
    temporal_logits = temporal_logits.cpu().detach().numpy()
    fusion_logits = np.mean(np.array([spatial_logits, temporal_logits]), axis=0)

    return fusion_logits

# Compute the top1 accuracy of predictions made by the network
def compute_accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    
    correct_predictions = 0
    for i, prediction in enumerate(predictions):
        if prediction == labels[i]:
            correct_predictions += 1

    return float(correct_predictions) / len(predictions)

# Compute the given top k accuracy of predictions made by the network
def compute_topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    target = target.cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_class_accuracy():
    return 0

# Save model to disk
def save_checkpoint(spatial_state, temporal_state, is_best_model, name, save_path):    

    checkpoint_path = f'{save_path}/{name}'
    
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    torch.save(spatial_state, f'{checkpoint_path}/spatial')
    torch.save(temporal_state, f'{checkpoint_path}/temporal')
    
    if is_best_model:
        shutil.copyfile(f'{checkpoint_path}/spatial', f'{checkpoint_path}/spatial_best')
        shutil.copyfile(f'{checkpoint_path}/temporal', f'{checkpoint_path}/temporal_best')


# Copy frames from dataset to prepare for bounding box placement
def copy_frames_to_output(model_output_path, dataset_path, videos):
    for video in tqdm(videos, desc="Copying video frames", leave=False, unit="video"):
        if not os.path.exists(f'{model_output_path}/{video}'):
            os.makedirs(f'{model_output_path}/{video}')
        
        files = glob.glob(f'{dataset_path}/frames/rgb/{video}/*.jpg')
        no_of_files = len(files)

        for file_path in tqdm(files, leave=False, unit="frame"):
            shutil.copy(file_path, f'{model_output_path}/{video}')
        

# Draw bounding boxes with prediction labels onto frames
def draw_bounding_boxes(args, predictions, classes, colours):
    annotations_path = f'{args.dataset_path}/annotations'

    # Draw bounding boxes on frames
    for video in tqdm(predictions.keys(), desc="Drawing bounding boxes", leave=False, unit="video"):
        for annotation in tqdm(predictions[video], leave=False):
            
            ape_id = annotation['ape_id']
            label = annotation['label']
            prediction = annotation['prediction']
            start_frame = annotation['start_frame']

            # Draw for n following frames
            for j in range(0, args.optical_flow):
                image = 0

                frame_no = start_frame + j

                ape = get_ape_by_id(annotations_path, video, frame_no, ape_id)
                ape_coords = get_ape_coordinates(ape)
                ape_species = get_species(ape)

                # Define bounding box text and colour
                label_text = f'{ape_id}:{classes[prediction]}'
                colour = colours[list(colours.keys())[ape_id]]

                # Show visually that prediction is incorrect
                if label != prediction:
                    colour = colours['red']
                    label_text = f'{label_text} != {classes[label]}'

                image_file_path = f'{args.output_path}/{args.name}/{video}/{video}_frame_{frame_no}.jpg'
                image = cv2.imread(image_file_path)

                top_left = (int(ape_coords[0]), int(ape_coords[1]))
                bottom_right = (int(ape_coords[2]), int(ape_coords[3]))

                # Draw main bounding box
                image = cv2.rectangle(image, top_left, bottom_right, colour, 2)
                
                # Get size of label text in order to create background rectangle to fit
                font = ImageFont.truetype("SF-Bold.otf", 14)  
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
def stitch_videos(model_output_path, dataset_path, predictions):
    for i, video in enumerate(tqdm(predictions.keys(), desc="Stitching videos", leave=False, unit="video")):
        img_array = []
    
        no_of_files = len(glob.glob(f'{dataset_path}/frames/rgb/{video}/*.jpg'))

        size = 0 
        for i in range(1, no_of_files + 1):
            filename = f'{model_output_path}/{video}/{video}_frame_{i}.jpg'

            if not os.path.exists(filename):
                break

            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        
        out = cv2.VideoWriter(f'{model_output_path}/{video}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


# Compute confusion matrix from labels and predictions
def compute_confusion_matrix(predictions_dict, classes, output_path):

    labels = []
    predictions = []
    for video in predictions_dict.keys():
        for annotation in predictions_dict[video]:
            labels.append(annotation['label'])
            predictions.append(annotation['prediction'])

    existing_classes = []

    for i in range(0, len(classes)):
        if i in set(labels):
            existing_classes.append(classes[i])
 
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(20,20))
    plot_confusion_matrix(cnf_matrix, classes=existing_classes, normalise=True, title='Normalised confusion matrix')

    plt.savefig(f'{output_path}/confusion_matrix.png', bbox_inches = "tight")
    # plt.show()


# Plots the confusion matrix
def plot_confusion_matrix(cm, classes, normalise=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalised Confusion Matrix")
    # else:
    #     print('Unnormalised Confusion Matrix')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30, pad=30)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)


    plt.tight_layout()
    plt.autoscale()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

def zip_videos(model_output_path, name):
    file_paths = []
    file_names = []

    for root, directories, files in os.walk(model_output_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
            file_names.append(os.path.basename(filepath))

    with ZipFile(f'{model_output_path}/{name}.zip','w') as zip: 
        for i, file in enumerate(tqdm(file_paths, desc="Zipping videos", leave=False, unit="video")): 
            zip.write(file, arcname=file_names[i]) 

def upload_videos(model_output_path, name, bucket):
     
    s3 = boto3.client('s3')
    
    # Validate AWS credentials 
    try:
        response = s3.list_buckets()
        print(f"AWS account credentials found. Uploading output to S3 bucket {bucket}")
    except ClientError as e:
        print("AWS account credentials not found. Skipping upload.")

    s3.upload_file(f'{model_output_path}/{name}.zip', bucket, f'{name}.zip')
    object_acl = s3.put_object_acl(ACL='public-read', Bucket=bucket, Key=f'{name}.zip')
    response = s3.generate_presigned_url('get_object', Params = {'Bucket': bucket, 'Key': f'{name}.zip', 'ResponseExpires': 3600})
    return response