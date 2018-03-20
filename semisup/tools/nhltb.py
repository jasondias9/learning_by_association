import os
import sys
from . import data_dirs
import cv2
import numpy as np
from tensorflow import convert_to_tensor
import scipy.misc


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

    for lab in dirs:
        curr = DATADIR + partition + '/' + lab
        for f in os.listdir(curr):
            if f.endswith('.jpeg'):
                imgdata = np.asarray(scipy.misc.imresize(cv2.imread(curr+'/'+f), (HEIGHT, WIDTH)))
                X.append(imgdata)
                Y.append(int(lab))
                
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
