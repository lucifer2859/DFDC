import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from ranger import Ranger

import sys
sys.path.insert(0, "/home/dchen/DFDC/RAdam")

from radam import RAdam

device = torch.device("cuda")
torch.cuda.set_device(10)

### Setup Data

df_trains = []
for i in range(47):
    df_trains.append(pd.read_json('/home/dchen/DFDC/deepfake/metadata' + str(i) + '.json'))

val_nums = [47, 48, 49]
df_vals = []
for val_num in val_nums:
    df_vals.append(pd.read_json('/home/dchen/DFDC/deepfake/metadata' + str(val_num) + '.json'))

nums = list(range(len(df_trains) + 1))
LABELS = ['REAL','FAKE']


def get_path(num, x):
    num = str(num)
    if len(num) == 2:
        path = '/home/dchen/DFDC/deepfake/DeepFake' + num + '/DeepFake' + num + '/' + x.replace('.mp4', '') + '.jpg'
    else:
        path = '/home/dchen/DFDC/deepfake/DeepFake0' + num + '/DeepFake0' + num + '/' + x.replace('.mp4', '') + '.jpg'
    if not os.path.exists(path):
        raise Exception
    return path

paths = []
y = []

for df_train, num in tqdm(zip(df_trains, nums), total=len(df_trains)):
    images = list(df_train.columns.values)
    for x in images:
        try:
            paths.append(get_path(num, x))
            y.append(LABELS.index(df_train[x]['label']))
        except Exception as err:
            # print(err)
            pass

val_paths = []
val_y = []
for df_val, num in tqdm(zip(df_vals,val_nums), total=len(df_vals)):
    images = list(df_val.columns.values)
    for x in images:
        try:
            val_paths.append(get_path(num, x))
            val_y.append(LABELS.index(df_val[x]['label']))
        except Exception as err:
            # print(err)
            pass

def read_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def shuffle(X, y):
    new_train = []
    for m, n in zip(X, y):
        new_train.append([m, n])

    random.shuffle(new_train)

    X, y = [], []
    for x in new_train:
        X.append(x[0])
        y.append(x[1])

    return X, y


import random

def get_random_sampling(paths, y, val_paths, val_y):
    real = []
    fake = []
    for m, n in zip(paths, y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)

    paths, y = [], []
    for x in real:
        paths.append(x)
        y.append(0)
    for x in fake:
        paths.append(x)
        y.append(1)

    real = []
    fake = []
    for m, n in zip(val_paths, val_y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)

    val_paths, val_y = [], []
    for x in real:
        val_paths.append(x)
        val_y.append(0)
    for x in fake:
        val_paths.append(x)
        val_y.append(1)

    X = []
    for img in tqdm(paths):
        X.append(read_img(img))
    val_X = []
    for img in tqdm(val_paths):
        val_X.append(read_img(img))
    
    
    # Balance with ffhq dataset
    ffhq = os.listdir('/home/dchen/DFDC/ffhq-face-data-set/thumbnails128x128')
    X_ = []
    for file in tqdm(ffhq):
        im = read_img(f'/home/dchen/DFDC/ffhq-face-data-set/thumbnails128x128/{file}')
        im = cv2.resize(im, (150,150))
        X_.append(im)
    # random.shuffle(X_)
    
    for i in range(4850):
        val_X.append(X_[i])
        val_y.append(0)
    
    del X_[0:4850]

    for i in range(65150):
        X.append(X_[i])
        y.append(0)

    X, y = shuffle(X, y)
    val_X, val_y = shuffle(val_X, val_y)

    return X, val_X, y, val_y


### Dataset

from torch.utils.data import Dataset, DataLoader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class ImageDataset(Dataset):
    def __init__(self, X, y, training=True, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.X[idx]

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        
        img = np.rollaxis(img, 2, 0)

        labels = self.y[idx]
        labels = np.array(labels).astype(np.float32)
        return [img, labels]


### Model

from pytorchcv.model_provider import get_model
model = get_model("xception", pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out

class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)
  
    def forward(self, x):
        x = self.base(x)
        return self.h1(x)

model = FCN(model, 2048)

# from torchtoolbox.tools import summary

# model.to(device)
# summary(model, torch.rand((1, 3, 150, 150)).cuda())


### Train Functions

def criterion1(pred1, targets):
    l1 = F.binary_cross_entropy(F.sigmoid(pred1), targets)
    return l1

def train_model(epoch, optimizer, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    for i, (img_batch, y_batch) in enumerate(t):
        img_batch = img_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        optimizer.zero_grad()

        out = model(img_batch)
        loss = criterion1(out, y_batch)

        total_loss += loss
        t.set_description(f'Epoch {epoch + 1} / {n_epochs}, LR: %6f, Loss: %.4f' % (optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))

        if history is not None:
            history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def evaluate_model(epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred = []
    real = []
    with torch.no_grad():
        for img_batch, y_batch in val_loader:
            img_batch = img_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            o1 = model(img_batch)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            for j in o1:
                pred.append(F.sigmoid(j))
            for i in y_batch:
                real.append(i.data.cpu())
    
    pred = [p.data.cpu().numpy() for p in pred]
    pred2 = pred
    pred = [np.round(p) for p in pred]
    pred = np.array(pred)
    acc = sklearn.metrics.recall_score(real, pred, average='macro')

    real = [r.item() for r in real]
    pred2 = np.array(pred2).clip(0.1, 0.9)
    kaggle = sklearn.metrics.log_loss(real, pred2)

    loss /= len(val_loader)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    if scheduler is not None:
        scheduler.step(loss)

    print(f'Dev loss: %.4f, Acc: %.6f, Kaggle: %.6f'%(loss,acc,kaggle))
    
    return loss


### Dataloaders

X, val_X, y, val_y = get_random_sampling(paths, y, val_paths, val_y)

print('There are ' + str(y.count(1)) + ' fake train samples')
print('There are ' + str(y.count(0)) + ' real train samples')
print('There are ' + str(val_y.count(1)) + ' fake val samples')
print('There are ' + str(val_y.count(0)) + ' real val samples')

import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression, CoarseDropout
train_transform = albumentations.Compose([
                                          ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                          HorizontalFlip(p=0.2),
                                          RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                                          MotionBlur(p=.2),
                                          GaussNoise(p=.2),
                                          JpegCompression(p=.2, quality_lower=50),
                                          # CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, always_apply=False, p=0.5),
                                          Normalize()
])
val_transform = albumentations.Compose([
                                          Normalize()
])

train_dataset = ImageDataset(X, y, transform=train_transform)
val_dataset = ImageDataset(val_X, val_y, transform=val_transform)


### Train

import gc

history = pd.DataFrame()
history2 = pd.DataFrame()

torch.cuda.empty_cache()
gc.collect()

best = 1e10
n_epochs = 20
batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)
optimizer = Ranger(model.parameters(), lr=0.0004)
# optimizer = RAdam(model.parameters(), lr=0.00035)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()

    train_model(epoch, optimizer, scheduler=None, history=history)
    
    loss = evaluate_model(epoch, scheduler=scheduler, history=history2)
    
    if loss < best:
      best = loss
      print(f'Saving best model...')
      torch.save(model.state_dict(), f'/home/dchen/DFDC/models/model_v22.pth')

# history2.plot()
# plt.savefig('evaluate_model.jpg')