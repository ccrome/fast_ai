from __future__ import division,print_function
import sys
import argparse
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import utils; reload(utils)
from utils import plots
from vgg16 import Vgg16
import keras.models 

parser = argparse.ArgumentParser()
parser.add_argument("root_path", type=str, help="Root path of your data directory")
parser.add_argument("--train-path", type=str, default="train", help="sub directory of root_path that contains the training data, defaults to 'train'")
parser.add_argument("--valid-path", type=str, default="valid", help="sub directory of root_path that contains the validation data, defaults to 'valid'")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size -- number of images to run at once")


def write_epoch(path, ep):
    epochfile = os.sep.join([path, "epoch.txt"])
    f = open(epochfile, "w")
    f.write("%d\n" % ep)
    f.close()
    
def get_epoch(path):
    epochfile = os.sep.join([path, "epoch.txt"])
    ep = None
    if (os.path.isfile(epochfile)):
        f = open(epochfile)
        l = f.readline()
        ep = int(l)
        f.close()
    if (ep == None):
        write_epoch(path, 0)
        return 0
    return ep
    
args = parser.parse_args()
train_path = os.sep.join([args.root_path, args.train_path])
valid_path = os.sep.join([args.root_path, args.valid_path])
weights_root = os.sep.join([args.root_path, 'weights_%03d'])


vgg = Vgg16()
batches     = vgg.get_batches(train_path, batch_size=args.batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=args.batch_size*2)

epoch = get_epoch(args.root_path)
model_filename = os.sep.join([args.root_path, "model.dat"])
if (epoch == 0):
    # Don't load weights, just use the raw vgg weights
    vgg.model.summary()
    vgg.finetune(batches)
    vgg.model.summary()
#    f = open(model_filename, "w")
#    f.write(vgg.model.to_json())
#    f.close()

    print ("**** Running the first epoch, not loading any weights")
else:
    print ("**** Running epoch %d" % (epoch))
    weights_fn = weights_root % epoch
    model = keras.models.load_model(weights_fn)
    vgg.model = model
    vgg.model.summary()
    print ("**** Loading weights from %s" % weights_fn)
    sys.stdout.flush()
    model.load_weights(weights_fn)
    exit()

#vgg.fit(batches, val_batches, nb_epoch=1)

epoch = epoch + 1
write_epoch(args.root_path, epoch)
print ("**** Saving weights to %s" % (weights_root % epoch))
model = vgg.model
help(model)
model.save(weights_root % epoch)
mm = keras.models.load_model(weights_root % epoch)


