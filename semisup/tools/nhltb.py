import os
import sys
from . import data_dirs
import cv2
import numpy as np
from tensorflow import convert_to_tensor


DATADIR = data_dirs.nhltb
WIDTH = 256
HEIGHT = 144


NUM_LABELS = 2
IMAGE_SHAPE = [HEIGHT, WIDTH, 3]

#return a tuple of images and lables
def get_data(name):
    if name in ['train', 'unlabeled']:
        print('[INFO]   Reading training images...')
        return load_data(DATADIR, 'train')
    elif name == 'test':
        print('[INFO]   Reading testing images...')
        return load_data(DATADIR, 'test')


def load_data(root, partition):

    dirs = next(os.walk(DATADIR+partition))[1]
    X = []
    Y = []
    unlab = []
    for lab in dirs:
        curr = DATADIR + partition + '/' + lab
        for f in os.listdir(curr):
            if f.endswith('.jpeg') and int(lab) > -1:
                imgdata = np.asarray(cv2.resize(cv2.imread(curr+'/'+f), (WIDTH, HEIGHT)))
                X.append(imgdata)
                Y.append(int(lab))
            elif f.endswith('.jpeg'):
                imgdata = np.asarray(cv2.resize(cv2.imread(curr+'/'+f), (WIDTH, HEIGHT)))
                unlab.append(imgdata)
    X = np.array(X)
    Y = np.array(Y)
    unlab = np.array(unlab)
    return X, Y, unlab
