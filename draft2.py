# Libraries
import os
from os.path import join
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import cv2
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
# from albumentations.pytorch import ToTensorV2



DATA_PATH = r'd:\05.temp\sartorius-cell-instance-segmentation'
SAMPLE_SUBMISSION = join(DATA_PATH, 'train')
TRAIN_CSV = join(DATA_PATH, 'train.csv')
TRAIN_PATH = join(DATA_PATH, 'train')
TEST_PATH = join(DATA_PATH, 'test')
df_train = pd.read_csv(TRAIN_CSV)


def initialize_seeds(seed):

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    # print(s)
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
    return np.array(mask)

# Training dataset & data loader
class CellDataset(Dataset):
    def __init__(self, df: pd.core.frame.DataFrame, train: bool):
        global TRAIN_PATH, df_train
        self.IMAGE_RESIZE = (224, 224)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)
        self.df = df
        self.base_path = TRAIN_PATH
        self.gb = self.df.groupby('id')
        self.transforms = Compose([Resize(self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]),
                                   Normalize(mean=self.RESNET_MEAN, std=self.RESNET_STD, p=1),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5)])

        # Split train and val set
        all_image_ids = np.array(df_train.id.unique())
        print('len(all_image_ids)= ', len(all_image_ids))
        np.random.seed(42)
        iperm = np.random.permutation(len(all_image_ids))
        num_train_samples = int(len(all_image_ids) * 0.9)
        print('num_train_samples= ', num_train_samples)

        if train:
            self.image_ids = all_image_ids[iperm[:num_train_samples]]
        else:
            self.image_ids = all_image_ids[iperm[num_train_samples:]]

    def __getitem__(self, idx: int) -> dict:

        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)

        # Read image
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)

        # Create the mask
        mask = build_masks(df_train, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        # print(np.moveaxis(image,0,2).shape)
        return np.moveaxis(np.array(image), 2, 0), mask.reshape((1, self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)


# Unet Model
class DoubleConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inChannel, outChannel, 3, padding=1),
                                  nn.BatchNorm2d(outChannel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(outChannel, outChannel, 3, padding=1),
                                  nn.BatchNorm2d(outChannel),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, inChannel, outChannel):
        super(Down, self).__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                  DoubleConv(inChannel, outChannel))

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):

    def __init__(self, inChannel, outChannel):
        super(Up, self).__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(inChannel, inChannel // 2, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv = DoubleConv(inChannel, outChannel)

    def forward(self, x, skipX):
        # 转置卷积
        x = self.upsample(x)
        x = torch.cat((x, skipX), dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.donw1 = DoubleConv(3, 64)
        self.donw2 = Down(64, 128)
        self.donw3 = Down(128, 256)
        self.donw4 = Down(256, 512)
        self.donw5 = Down(512, 1024)
        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.oneMult = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (batchSize, channel, h, w)
        # 5次下采样，4次上采样，oneMult1*1卷积，不改变size改变通道数
        x1 = self.donw1(x)
        x2 = self.donw2(x1)
        x3 = self.donw3(x2)
        x4 = self.donw4(x3)
        x = self.donw5(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.oneMult(x)

        return x


# Train the network
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0): # 当验证集损失在连续7次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss




def valid(loader, model, criterion):
    model.eval()
    runningLoss = 0
    totalIou = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as tq:
            for i, (img, target) in enumerate(loader):
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = criterion(output, target)
                runningLoss += loss.item() * img.size(0)

                iou = computeIOU(output, target)
                totalIou += iou * img.size(0)
                tq.update(1)
            valid_curve.append(loss.item())

    return runningLoss / len(loader.dataset), totalIou / len(loader.dataset)


def computeIOU(output, mask):
    pred = torch.zeros(output.size()).cuda()
    pred[output > 0] = 1
    pred = pred.to(torch.uint8)
    mask = mask.to(torch.uint8)
    intersection = (pred & mask)
    union = (pred | mask)
    return torch.sum(intersection).item() / torch.sum(union).item()



def main():
    initialize_seeds(2021)

    print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} \
    Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    ds_train = CellDataset(df_train, train=True)
    dl_train = DataLoader(ds_train, batch_size=16, num_workers=2, pin_memory=True, shuffle=False)

    ds_val = CellDataset(df_train, train=False)
    dl_val = DataLoader(ds_val, batch_size=4, num_workers=2, pin_memory=True, shuffle=False)

    print('len dl_train= ', len(dl_train))

    # plot simages and mask from dataloader
    batch = next(iter(dl_train))
    images, masks = batch
    print(f"image shape: {images.shape},\nmask shape:{masks.shape},\nbatch len: {len(batch)}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(images[1][1])
    plt.title('Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(masks[1][0])
    plt.title('Mask')

    plt.subplot( 1, 3, 3)
    plt.imshow(images[1][1])
    plt.imshow(masks[1][0],alpha=0.2)
    plt.title('Both')
    plt.tight_layout()
    plt.show()

    # Define the network
    # put on GPU here if you have it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net.to(device);  # remove semi-colon to see net structure
    print(f"The device is {device}!!")

    model = Unet()
    model.to(device)

    # 保存整个模型
    # torch.save(model, path_model)

    # 保存模型参数
    # net_state_dict = model.state_dict()
    # torch.save(net_state_dict, path_state_dict)

    # 损失函数&优化器
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_curve = list()
    valid_curve = list()

    lowestLoss = None
    # checkpoint_interval = 5
    patience = 7
    early_stopping = EarlyStopping(patience, verbose=True)

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    def train(loader, model, criterion, optimizer):

        model.train()
        runningLoss = 0
        with tqdm(total=len(dl_train)) as tq:
            for i, (img, target) in enumerate(loader):
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                runningLoss += loss.item() * img.size(0)
                tq.update(1)
            train_curve.append(loss.item())

        return runningLoss / len(loader.dataset)

    for epoch in range(50):  # 以设置大一些，希望通过 early stopping 来结束模型训练
        trainLoss = train(dl_train, model, criterion, optimizer)
        validLoss, iou = valid(dl_val, model, criterion)
        print('Epoch ：{}'.format(epoch))
        print('Train Loss : {}    Valid Loss : {}'.format(trainLoss, validLoss))
        print('IOU : {}'.format(iou))

        if lowestLoss is None or validLoss < lowestLoss:
            lowestLoss = validLoss
            torch.save({'model_state_dict': model.state_dict(),
                        'trainLoss': trainLoss,
                        'validLoss': validLoss,
                        'optimizer': optimizer,
                        'iou': iou},
                       f'./checkpoint.pth')

        early_stopping(validLoss, model)
        # 若满足早停要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(dl_train)
    # valid_x = np.arange(1, len(valid_curve)+1) * train_iters # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(train_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

    # checkpoint为字典型
    # model.load_state_dict(torch.load('./checkpoint.pth'))
    torch.load('./checkpoint.pt')


if __name__ == '__main__':
    main()
