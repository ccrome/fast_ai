from glob import glob
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root_path", type=str, help="Root path of your data directory")
parser.add_argument("--train-path", type=str, default="train", help="sub directory of root_path that contains the training data, defaults to 'train'")
parser.add_argument("--valid-path", type=str, default="valid", help="sub directory of root_path that contains the validation data, defaults to 'valid'")
parser.add_argument("--valid-percent", type=float, default=10.0, help="Percent of files to move from the training path to the validation path, default = 10 (%%).  Enter as a percent, not as a fraction")
parser.add_argument("-n", action='store_true', help="Dry run, don't actually move the files, but show what would be done")

args = parser.parse_args()

train_path = os.sep.join([args.root_path, args.train_path])
valid_path = os.sep.join([args.root_path, args.valid_path])

def create_validation_files(train_path, valid_path, num_to_move=None, percent_to_move=None, dry_run = False):
    # Create validation set by randomly choosing a set of files from files
    # and moving them to a 'valid' directory with the same hierarchy
    # returns the new list of files in the test directory, and the list of files
    # in the validation directory

    globdir = os.sep.join([train_path, "*", "*.jpg"])
    files = np.random.permutation(glob(globdir))

    if (percent_to_move != None):
        num_to_move = int(len(files) * percent_to_move/100.)
    if (num_to_move == None):
        raise Exception("You must specify either a number of files to move (num_to_move) or percent of files to move (percent_to_move)")
    
    moved_files = files[:num_to_move]
    kept_files = files[num_to_move:]
    print "Found %d files" % len(files)
    print "Moving %d files " % len(moved_files)

    for fn in moved_files:
        fna = fn.split(os.sep)
        fnn = os.sep.join(fna[-2:])
        new_fn = os.sep.join([valid_path, fnn])
        new_dir = os.path.dirname(new_fn)
        if (dry_run):
            print "move %s to %s" % (fn, new_fn)
        else:
            try:
                os.makedirs(new_dir)
            except OSError:
                pass
            os.rename(fn, new_fn)
    return moved_files, kept_files

create_validation_files(train_path, valid_path, percent_to_move = args.valid_percent, dry_run=args.n)

#
#
#try:
#    os.mkdir("valid")
#except Exception:
#    pass
#
#    
#
#print len(files)
#print files[:5]
