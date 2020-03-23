# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

test_dir = "/home/dchen/DFDC/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

input_size = 299
frames_per_video = 2

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))

    return preprocessed_image

def test_full_image_network(video_path, model, batch_size):
    print('Starting: {}'.format(video_path))
    
    # Read
    reader = cv2.VideoCapture(video_path)

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Frame numbers and length of output video
    frame_num = 0 
    preprocessed_images = [] 
    post_function = nn.Softmax(dim=1)

    st = time.time()

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        # Image size
        height, width = image.shape[:2]

        # Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            preprocessed_images.append(preprocess_image(cropped_face))

        if frame_num >= batch_size:
            break

    dlib_elapsed = time.time() - st
    print("Dlib elapsed %f sec. Average per frame: %f sec." % (dlib_elapsed, dlib_elapsed / batch_size))

     # Actual prediction using our model
    if len(preprocessed_images) > 0:
        data_x = torch.stack(preprocessed_images, 0).cuda(1)

        # print(torch.max(data_x))
        # print(torch.min(data_x))
        # print(torch.mean(data_x))

        y_pred = model(data_x)
        y_pred = post_function(y_pred)

        FAKE_prob = y_pred[:len(preprocessed_images)].mean(dim=0).detach().cpu().numpy()[1]

        xception_elapsed = time.time() - dlib_elapsed - st
        print("Xception elapsed %f sec. Average per frame: %f sec." % (xception_elapsed, xception_elapsed / batch_size))

        return FAKE_prob
    
    return 0.5

# Load model
model, *_ = model_selection(modelname='xception', num_out_classes=2)

model_path = '/home/dchen/DFDC/faceforensics++_models_subset/face_detection/xception/all_raw.p'

if model_path is not None:
    model = torch.load(model_path)
    print('Model found in {}'.format(model_path))
else:
    print('No model found, initializing random model.')

model = model.cuda(1)

speed_test = True  # you have to enable this manually
if speed_test:
    start_time = time.time()
    speedtest_videos = test_videos[:20]
    for filename in speedtest_videos:
        y_pred = test_full_image_network(join(test_dir, filename), model, batch_size=frames_per_video)
        print(y_pred)
    elapsed = time.time() - start_time
    print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))
else:
    predictions = []
    
    for filename in test_videos:
        predictions.append(test_full_image_network(join(test_dir, filename), model, batch_size=frames_per_video))
    
    submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
    submission_df.to_csv("submission.csv", index=False)