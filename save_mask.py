'''
python 3.9 - kaggle sartorius
save mask - не работает - сохраняет черные квадраты

Version: 0.1
Author: Andrej Marinchenko
Date: 2021-12-16
'''

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2  # computer vision library
import matplotlib.pyplot as plt  # collection of command style functions creates a figure, creates a plotting area in
# a figure, plots some lines in a plotting area

# df_train = pd.read_csv("../input/sartorius-cell-instance-segmentation/train.csv")
df_train = pd.read_csv(r'd:\05.temp\sartorius-cell-instance-segmentation\train.csv')
print(df_train)



def rle_decode(mask_rle, shape, color=1):  # function to convert tabular mask data to image
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return
    color: color for the mask
    Returns numpy array (mask)

    '''
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]

    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)

    for start, end in zip(starts, ends):
        img[start: end] = color

    return img.reshape(shape)  # Gives a new shape to an array without changing its data


def plot_masks(image_id, colors=False):  # mask inference function
    # at the input of the function, we indicate the number of the image and whether to display the mask in color
    # (colors=True) or  (colors=False)
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()  # by the image number, we take all the
    # value  in the column "annotation" which contain all the data on our mask

    if colors:  # (colors=True)
        mask = np.zeros((520, 704, 3))  # Return a new array of given shape and type, filled with zeros. Set the size
        #  and color scheme in rgb
        for label in labels:  # we go through all the data by masks (in the column "annotation")
            mask += rle_decode(label, shape=(520, 704, 3), color=np.random.rand(3))  # create an array of our masks
            # processed function rle_decode
    else:  # (colors=False)
        mask = np.zeros((520, 704, 1))  # Return a new array of given shape and type, filled with zeros. Set the size
        #  and color scheme in one color
        for label in labels:  # we go through all the data by masks (in the column "annotation")
            mask += rle_decode(label, shape=(520, 704, 1))  # create an array of our masks processed function rle_decode
    mask = mask.clip(0, 1)  # Clip (limit) the values in an array. Given an interval, values outside the interval are
    #  clipped to the interval edges.
    # if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.



    cv2.imwrite(fr"d:\05.temp\sartorius-cell-instance-segmentation\mask\{image_id}.png", mask)




# plot_masks("0030fd0e6378", colors=True)
plot_masks("0140b3c8f445", colors=False)