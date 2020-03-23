# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time
from os.path import join
import cv2
import torch
import torch.nn as nn

from network.models import model_selection

test_dir = "/home/dchen/DFDC/test_videos/"
test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])

gpu = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, "/home/dchen/DFDC/blazeface-pytorch")
sys.path.insert(0, "/home/dchen/DFDC/deepfakes-inference-demo")

from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/home/dchen/DFDC/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/home/dchen/DFDC/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)

from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 16

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

input_size = 299

from torchvision.transforms import Normalize

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

normalize_transform = Normalize(mean, std)

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def test_full_image_network(video_path, model, batch_size):
    print('Starting: {}'.format(video_path))
    
    # Find the faces for N frames in the video.
    faces = face_extractor.process_video(video_path)

    # Only look at one face per frame.
    face_extractor.keep_only_best_face(faces)

    post_function = nn.Softmax(dim=1)

    if len(faces) > 0:
        x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

        # If we found any faces, prepare them for the model.
        n = 0
        for frame_data in faces:
            for face in frame_data["faces"]:
                # Resize to the model's required input size.
                # We keep the aspect ratio intact and add zero
                # padding if necessary.                    
                resized_face = isotropically_resize_image(face, input_size)
                resized_face = make_square_image(resized_face)

                if n < batch_size:
                    x[n] = resized_face
                    n += 1

        if n > 0:
            x = torch.tensor(x, device=gpu).float()

            # Preprocess the images.
            x = x.permute((0, 3, 1, 2))

            for i in range(len(x)):
                x[i] = normalize_transform(x[i] / 255.)

            # print(torch.max(x))
            # print(torch.min(x))
            # print(torch.mean(x))

            # Make a prediction, then take the average.
            with torch.no_grad():
                y_pred = model(x)
                y_pred = post_function(y_pred)
                FAKE_prob = y_pred[:n].mean(dim=0).detach().cpu().numpy()[1]

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

model = model.to(gpu)

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