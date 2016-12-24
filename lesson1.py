

# %matplotlib inline

from __future__ import division,print_function

path = "data/redux/"

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import utils; reload(utils)
from utils import plots
batch_size=64
from vgg16 import Vgg16
vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)


run_epoch = 0

if run_epoch:
    vgg.finetune(batches)
    vgg.fit(batches, val_batches, nb_epoch=1)
    vgg.model.save_weights("my_weights")
else:
    vgg.model.load_weights("my_weights")


batches = vgg.get_batches(path+'test', batch_size = batch_size);
imgs,labels = next(batches)

#plots(imgs[:4], titles=labels[:4])

vgg.classes = ["cat", "dog"]
results = vgg.predict(imgs[:4], True)
maxes = np.argmax(results, axis=1)

print (results)



