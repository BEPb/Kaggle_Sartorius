'''
python 3.9 - kaggle sartorius - Epoch: 10 - Train Loss 1.2253

Version: 0.0
Author: Andrej Marinchenko
Date: 2021-12-16
'''

import gc
import os
import pdb
import time
import glob
import sys
import cv2
import imageio
import joblib
import math
import random
import math

import numpy as np
import pandas as pd

# import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 18})
plt.style.use('fivethirtyeight')

# import seaborn as sns
# import matplotlib
# from dask import bag, diagnostics
# from mpl_toolkits.mplot3d import Axes3D
#
# from termcolor import colored

# from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler

import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp

from sklearn.model_selection import KFold

from albumentations import (HorizontalFlip, VerticalFlip,
                            ShiftScaleRotate, Normalize, Resize,
                            Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2

import warnings
warnings.simplefilter('ignore')

# Activate pandas progress apply bar
tqdm.pandas()


# import parallelTestModule

# if __name__ == '__main__':
    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)


class config:
    # DIRECTORY_PATH = "../input/sartorius-cell-instance-segmentation"
    DIRECTORY_PATH = r"d:\05.temp\sartorius-cell-instance-segmentation"
    TRAIN_CSV = DIRECTORY_PATH + r"\train.csv"
    TRAIN_PATH = DIRECTORY_PATH + r"\train"
    TEST_PATH = DIRECTORY_PATH + r"\test"
    TRAIN_SEMI_SUPERVISED_PATH = DIRECTORY_PATH + r"\train_semi_supervised"

    SEED = 42

    RESNET_MEAN = (0.485, 0.456, 0.406)
    RESNET_STD = (0.229, 0.224, 0.225)

    # (336, 336)
    IMAGE_RESIZE = (224, 224)

    LEARNING_RATE = 5e-4
    EPOCHS = 10


def set_seed(seed=config.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

df_train = pd.read_csv(config.TRAIN_CSV)


def getImagePaths(path):
    """
    Функция объединения пути к каталогу с отдельными путями к изображениям
    параметры: path (строка) - Путь к каталогу
    возвращает: image_names (строка) - Полный путь к изображению
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in tqdm(filenames):
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

#Get complete image paths for train and test datasets
train_images_path = getImagePaths(config.TRAIN_PATH)
test_images_path = getImagePaths(config.TEST_PATH)
train_semi_supervised_path = getImagePaths(config.TRAIN_SEMI_SUPERVISED_PATH)


# U-Net Model

# utilites
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return mask

# Dataset Class
class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.base_path = config.TRAIN_PATH
        self.transforms = Compose([Resize(config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]),
                                   Normalize(mean=config.RESNET_MEAN, std=config.RESNET_STD, p=1),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5),
                                   ToTensorV2()])
        self.gb = self.df.groupby('id')
        self.image_ids = df.id.unique().tolist()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)
        mask = build_masks(df_train, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask.reshape((1, config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)

# Prepare Dataset
ds_train = CellDataset(df_train)
image, mask = ds_train[1]
image.shape, mask.shape

# plt.imshow(image[0], cmap='bone')
# plt.show()
# plt.imshow(mask[0], alpha=0.3)
# plt.show()


# Prepare Dataloader
dl_train = DataLoader(
    ds_train,
    batch_size=16,  # 64, 32,
    num_workers=4,
    pin_memory=True,
    shuffle=False
)

# Losses
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


 # Focal Loss¶
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

if __name__ == '__main__':
    # Model
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

    # Training
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    n_batches = len(dl_train)

    model.cuda()
    model.train()

    criterion = MixedLoss(10.0, 2.0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


    for epoch in range(1, config.EPOCHS + 1):
        print(f"Starting epoch: {epoch} / {config.EPOCHS}")
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dl_train):
            # Predict
            images, masks = batch
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)


            # Back prop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = running_loss / n_batches
        print(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f}")