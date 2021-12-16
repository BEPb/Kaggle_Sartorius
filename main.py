'''
 Sartorius - Torch - Classifier + Mask R-CNN
'''


import os
import time
import random
import collections

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Fix randomness
fix_all_seeds(2021)




# TRAIN_CSV = "../input/sartorius-cell-instance-segmentation/train.csv"
TRAIN_CSV = r"C:\Users\andre\Downloads\sartorius-cell-instance-segmentation\train.csv"
# TEST_PATH = "../input/sartorius-cell-instance-segmentation/test"
TEST_PATH = r"C:\Users\andre\Downloads\sartorius-cell-instance-segmentation\test"

# CLASSIFIER_CHK = "../input/sartorius-resnet-34-classifier-finetuned/resnet34-finetuned.bin"
CLASSIFIER_CHK = r"c:\Users\andre\Downloads\resnet34-finetuned\resnet34-finetuned.bin"
# MASK_RCNN_CHK = "../input/sartorius-starter-torch-mask-r-cnn/pytorch_model.bin"
MASK_RCNN_CHK = r"c:\Users\andre\Downloads\pytorch_model\pytorch_model.bin"

CELL_TYPES  = {0: 'shsy5y', 1: 'astro', 2: 'cort'}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)


# The maximum possible amount of predictions
# 539 is the 90% percentile of the cell_type with more instances per image
BOX_DETECTIONS_PER_IMG = 559

# load df
df_train = pd.read_csv(TRAIN_CSV)

# Simple statistics: number of instances per image per cell_type
# We will use the values from this analysis to decide the number of predicted individuals to generate for each image
# Простая статистика: количество экземпляров на изображение на тип ячейки
# Мы будем использовать значения из этого анализа, чтобы определить количество прогнозируемых лиц, которые нужно
# сгенерировать для каждого изображения.

df_instances = df_train.groupby(['id']).agg({'annotation': 'count', 'cell_type': 'first'})
df_instances = df_instances.groupby("cell_type")[['annotation']]\
                               .describe(percentiles=[0.1, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]).astype(int)\
                               .T.droplevel(level=0).T.drop(['count', '50%', 'std'], axis=1)
df_instances

# Trying with different strategies
df_instances['90%'].to_dict()

df_train['n_pixels'] = df_train.annotation.apply(lambda x: np.sum([int(e) for e in x.split()[1:][::2]]))
df_pixels = df_train.groupby("cell_type")[['n_pixels']].describe(percentiles=[0.02, 0.05, 0.1, 0.9, 0.95, 0.98])\
                    .astype(int).T.droplevel(level=0).T.drop(['count', '50%', 'std'], axis=1)
df_pixels

# Models
# Mask R-CNN from  Sartorius - Starter Torch Mask R-CNN [LB=0.270]
# The model is trained here and provided as a dataset
# It comes from version 28, the epoch 18 (which is the one that performed the best).

# Модели
# Маска R-CNN от Sartorius - Маска стартового фонаря R-CNN [LB = 0,270]
# Модель обучается здесь и предоставляется в виде набора данных.
# Это происходит из версии 28, эпохи 18 (которая показала лучшие результаты).


def get_pretrained_mask_cnn():
    # This is just a dummy value for the classification head
    NUM_CLASSES = 2

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False,
                                                               pretrained_backbone=False,
                                                               box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    model.load_state_dict(torch.load(MASK_RCNN_CHK, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


model = get_pretrained_mask_cnn()

# Classifier from Sartorius - Resnet34 Classifier
# Я по ошибке сохранил полную модель вместо dict состояния, поэтому загрузка выполняется без
# определения архитектуры, что довольно непонятно.
# Архитектура:
# from torchvision.models import resnet34
# m = resnet34(True)
# m.fc = nn.Linear(512, 3)

# Load the fine-tuned resnet34 classifier for cell_types
classifier = torch.load(CLASSIFIER_CHK, map_location=DEVICE)
classifier.to(DEVICE)
classifier.eval()


### Classifier utility functions
# Get the input of the classifier
# The process overlaps a bit with the Mask R-CNN preprocessing
# But they are different
def get_image_for_classifier(image_id):
    image_path = os.path.join(TEST_PATH, image_id + '.png')
    transforms = A.Compose([A.Resize(224, 224),
                       A.Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1),
                       ToTensorV2()])
    image = transforms(image=cv2.imread(image_path))['image']
    return image.unsqueeze(0).to(DEVICE)

# Assess the image_id cell_type with the classifier
def get_image_cell_type(classifier, image_id):
    img = get_image_for_classifier(image_id)
    with torch.no_grad():
        logits = classifier(img)[0]
        cell_type_idx = torch.argmax(logits).item()
    return CELL_TYPES[cell_type_idx]


### # Test Dataset

class CellTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_ids = [f[:-4] for f in os.listdir(self.image_dir)]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        return {'image': image, 'image_id': image_id}

    def __len__(self):
        return len(self.image_ids)


ds_test = CellTestDataset(TEST_PATH)

### Utility functions¶
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

### Prediction loop
df_instances.loc['cort', '10%']

# Mask refinement function
# The masks provided by Mask R-CNN have a probability in each of the pixels and we turn that into a mask thresholding it.
# A simple method to do this is to use one fixed number. That is MASK_THRESHOLD, which was fixed to 0.5 till now.
# Here we propose a refined method.
# The method changes the threshold value in border cases, making sure the number of pixels of the mask is not lower
# than the 5% percentile and not higher than the 95% percentile of the train data for the givel cell_type.

# Функция уточнения маски
# Маски, предоставляемые Mask R-CNN, имеют вероятность в каждом из пикселей, и мы превращаем это в пороговую маску.
# Простой способ сделать это - использовать одно фиксированное число. Это MASK_THRESHOLD, который до сих пор был
# установлен на 0,5.
# Здесь мы предлагаем усовершенствованный метод.
# Метод изменяет пороговое значение в пограничных случаях, следя за тем, чтобы количество пикселей маски было не ниже
# 5% -ного процентиля и не выше 95% -ного процентиля данных поезда для givel cell_type.

def refine_mask(mask, df_pixels, cell_type):
    # Minimum number of pixels:
    # The percentile 0.02 of the cell_type in the train set
    min_pixels = df_pixels.loc[cell_type, '2%']
    # Max number of pixels
    # The percentile 0.95 of the cell_type in the train set
    max_pixels = df_pixels.loc[cell_type, '98%']

    binary_mask = mask > MASK_THRESHOLD

    # If the mask is too small, make the condition less strict
    # increasing its size until it reaches a minimum number of pixels
    if binary_mask.sum() < min_pixels:
        for t in range(25):
            binary_mask = mask > (MASK_THRESHOLD - t * 0.02)
            if binary_mask.sum() > min_pixels:
                break

    # If the mask is too large, make the condition more strict
    # reducing its size until it has less than certain amount of pixels
    if binary_mask.sum() > max_pixels:
        for t in range(25):
            binary_mask = mask > (MASK_THRESHOLD + t * 0.02)
            if binary_mask.sum() < max_pixels:
                break

    return binary_mask


# Prediction
MIN_SCORE = 0.59

MASK_THRESHOLD = 0.5

submission = []
for sample in ds_test:
    img = sample['image']
    image_id = sample['image_id']

    # Get classifier prediction: cell_type
    cell_type = get_image_cell_type(classifier, image_id)

    # Given the cell_type, determine the numnber of instances to predict
    max_preds = df_instances.loc[cell_type, '99%']
    # min_preds = df_instances.loc[cell_type, '10%']

    # Get Mask R-CNN predictions
    with torch.no_grad():
        result = model([img.to(DEVICE)])[0]

    previous_masks = []
    for i, mask in enumerate(result["masks"]):

        score = result["scores"][i].cpu().item()

        # Predict at most the 90% number of instances per cell type
        if i >= max_preds:
            break

        # Minimum score required for instance to be kept
        if score < MIN_SCORE:
            break

        mask = mask.cpu().numpy()

        # See above "Mask refinement function"
        binary_mask = refine_mask(mask, df_pixels, cell_type)

        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)

        previous_masks.append(binary_mask)
        rle = rle_encoding(binary_mask)
        submission.append((image_id, rle))

    # Add empty prediction if no RLE was generated for this image
    all_images_ids = [image_id for image_id, rle in submission]
    if image_id not in all_images_ids:
        submission.append((image_id, ""))

df_sub = pd.DataFrame(submission, columns=['id', 'predicted'])
df_sub.to_csv("submission-lb0.28.csv", index=False)
df_sub.head()
