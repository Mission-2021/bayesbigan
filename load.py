import sys
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt
import pickle

from lib.data_utils import shuffle


def mnist():
    data_dir = os.path.join(os.environ["DATADIR"], "mnist")
    fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28))

    fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28))

    fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    
    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
  
    trX, teX, trY, teY = mnist()

    trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY

def cifar():
    data_dir = os.path.join(os.environ["DATADIR"], "cifar/cifar-10-batches-py")
    
    def process_batch(fn):
        fo = open(fn, 'rb')
        data_dict = pickle.load(fo, encoding="latin1")
        fo.close()
        raw = data_dict["data"]
        images = raw.reshape((-1, 3, 32, 32))

        return images, np.array(data_dict["labels"], dtype=np.int32)

    
    trX, trY = [], []
    for i in range(1, 6):
        batch_name = os.path.join(data_dir, "data_batch_%d" % i)
        print(batch_name)
        images, labels = process_batch(batch_name)
        trX.append(images)
        trY.append(labels)
    trX = np.concatenate(trX)
    trY = np.concatenate(trY)
    
    teX, teY = process_batch(os.path.join(data_dir, "test_batch"))
     
    return trX, teX, trY, teY







