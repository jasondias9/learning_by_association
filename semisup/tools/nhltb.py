import os
import sys
from . import data_dirs
import cv2
import numpy as np
from tensorflow import convert_to_tensor

DATADIR = data_dirs.nhltb

NUM_LABELS = 1
IMAGE_SHAPE = [720, 1280, 3]

#return a tuple of images and lables
def get_data(name):
    if name in ['train', 'unlabeled']:
        return load_data(DATADIR, 'train')
    elif name == 'test':
        return load_data(DATADIR, 'test')


def load_data(root, partition):
    X = []
    Y = []
    print('[INFO]   Reading files...')
    dirs = next(os.walk(DATADIR))[1]
    for lab in dirs:
        curr = DATADIR + lab
        for f in os.listdir(curr):
            imgdata = np.asarray(cv2.imread(curr+'/'+f))
            X.append(imgdata)
            Y.append(lab)
    return X, Y
