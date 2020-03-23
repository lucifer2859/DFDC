import os
import json
from os.path import join
import cv2
import matplotlib.pyplot as plt
import torch

train_sample_dir = "/home/dchen/DFDC/train_sample_videos/"
train_sample_videos = sorted([x for x in os.listdir(train_sample_dir) if x[-4:] == ".mp4"])

print(len(train_sample_videos))

cropped_faces_dir = "/home/dchen/DFDC/cropped_faces/blazeface/"
train_face_dir = join(cropped_faces_dir, 'train')
val_face_dir = join(cropped_faces_dir, 'val')

os.makedirs(train_face_dir, exist_ok=True)
os.makedirs(val_face_dir, exist_ok=True)

os.makedirs(join(train_face_dir, 'REAL'), exist_ok=True)
os.makedirs(join(train_face_dir, 'FAKE'), exist_ok=True)
os.makedirs(join(val_face_dir, 'REAL'), exist_ok=True)
os.makedirs(join(val_face_dir, 'FAKE'), exist_ok=True)

jsonlist = None
with open(join(train_sample_dir, 'metadata.json'),'r') as fp:
    jsonlist=json.load(fp)

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

frames_per_video = 200

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)

def crop_face_from_video(video_name):
    label = jsonlist[video_name]["label"]
    print('Starting: {}, label: {}'.format(video_name, label))

    video_path = join(train_sample_dir, video_name)
    video_fn = video_name.split('.')[0]

    faces = face_extractor.process_video(video_path)
    face_extractor.keep_only_best_face(faces)

    cropped_faces = []

    if len(faces) > 0:
        for frame_data in faces:
            for face in frame_data["faces"]:                        
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                cropped_faces.append(face)
        
        for cnt in range(len(cropped_faces)):
            if cnt % 5 == 0:
                cv2.imwrite(join(val_face_dir, label, video_fn + '_' + str(cnt + 1).rjust(3,'0') + '.jpg'), cropped_faces[cnt])
            else:
                cv2.imwrite(join(train_face_dir, label, video_fn + '_' + str(cnt + 1).rjust(3,'0') + '.jpg'), cropped_faces[cnt])

for video in train_sample_videos:  
    crop_face_from_video(video)

print('Train REAL image number: %d' % len(os.listdir(join(train_face_dir, 'REAL'))))
print('Train FAKE image number: %d' % len(os.listdir(join(train_face_dir, 'FAKE'))))
print('Validation REAL image number: %d' % len(os.listdir(join(val_face_dir, 'REAL'))))
print('Validation FAKE image number: %d' % len(os.listdir(join(val_face_dir, 'FAKE'))))
