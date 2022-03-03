# Code is copied and adapted from here
# https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/image_dataset.py
import numpy as np
from pathlib import Path
import imageio
import random
import logging
import re
import torch
from torch.utils.data import Dataset

def _natural_sort(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", x)]
    return sorted(l, key=key)

class InputError(Exception):
    def __init__(self, message):
        self.message = message
        
def avocado_classification(gt_fname):
    low_thr = 10**-6
    high_thr = 10**-3
    gt = np.loadtxt(gt_fname, skiprows=1, delimiter=",")
    air_ratio = gt[:,4] / gt[:,1:].sum(axis=1)
    y = np.where(air_ratio > low_thr, 1, 0)
    y[air_ratio > high_thr] = 2
    return y

def playdoh_classification(gt_fname):
    gt = np.loadtxt(gt_fname, skiprows=1, delimiter=",")
    stone_pixels = gt[:,2]
    y = np.where(stone_pixels > 0, 1, 0)
    return y

class ImageStack(object):
    def __init__(self, path_specifier, *, collapse_channels=False, labels=None):
        super(ImageStack, self).__init__()
        path_specifier = Path(path_specifier).expanduser().resolve()
        self.path_specifier = path_specifier
        self.collapse_channels = collapse_channels
        self.labels = labels

        self.paths = ImageStack.find_images(path_specifier)

    def find_images(path_specifier):
        path_specifier = Path(path_specifier)
        if path_specifier.name and "*" in path_specifier.name:
            paths = path_specifier.parent.glob(path_specifier.name)
        elif path_specifier.is_file():
            paths = [path_specifier]
            logging.warning("Image stack consists of single file {}".format(path_specifier))
        elif path_specifier.is_dir():
            paths = path_specifier.glob("*")
        else:
            paths = []
        paths = [str(p) for p in paths]
        paths = _natural_sort(paths)
        if len(paths) == 0:
            logging.warning("Image stack is empty for path specification {}".format(path_specifier))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = imageio.imread(path)
        except Exception as e:
            raise logging.exception("Could not read image from {}".format(path))

        if img.ndim != 2:
            raise InputError("Tif image is supposed to have only one channel: {}".format(path))

        img = torch.from_numpy(img)

        return img
    
class ImageDatasetTransformable(Dataset):
    '''Adaptation of Image Dataset from msd_pytorch with standard augmentation (random crop, flips, rotation).
    All augmentation are implemented using standard tensor operations, so it should work with torchvision 0.2.2
    '''
    def __init__(self, input_path_specifier, target_fname, sample_type, *,
                 random_crop=False, padding=20, crop_shape=(380,478),  vertical_flip=False, horizontal_flip=False, rotate=False):
        super(ImageDatasetTransformable, self).__init__()
        self.input_path_specifier = input_path_specifier
        self.random_crop = random_crop
        self.padding = padding
        self.crop_shape = crop_shape
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.rotate = rotate

        self.input_stack = ImageStack(input_path_specifier)
        if sample_type == "playdoh":
            self.target_labels = playdoh_classification(target_fname)
        elif sample_type == "avocado":
            self.target_labels = avocado_classification(target_fname)
        else:
            raise InputError("Unknown sample type. Got {}".format(sample_type))
        
        if len(self.input_stack) != len(self.target_labels):
            raise InputError("Number of inputs and target labels does not match. Got {} inputs and {} target labels.".format(
                  len(self.input_stack), len(self.target_labels) ))

    def __len__(self):
        return len(self.input_stack)

    def __getitem__(self, i):
        inp = self.input_stack[i]
        tg = self.target_labels[i]
        
        if self.horizontal_flip:
            if random.random() > 0.5:
                inp = torch.flip(inp, [1])
        if self.vertical_flip:
            if random.random() > 0.5:
                inp = torch.flip(inp, [0])
        if self.rotate:
            if random.random() > 0.5:
                num_rotations = random.randint(0,3)
                inp = torch.rot90(inp, num_rotations, [0, 1])
                if num_rotations in (1, 3):
                    #restore original shape
                    rot_shape = inp.shape
                    if rot_shape[1] > rot_shape[0]:
                        diff = (rot_shape[1] - rot_shape[0]) // 2
                        pad = (0, 0, diff+1, diff+1)
                        inp = torch.nn.functional.pad(inp, pad, "constant", 0)
                        inp = inp[0:rot_shape[1],diff:diff+rot_shape[0]]
                    if rot_shape[0] > rot_shape[1]:
                        diff = (rot_shape[0] - rot_shape[1]) // 2
                        pad = (diff+1, diff+1, 0, 0)
                        inp = torch.nn.functional.pad(inp, pad, "constant", 0)
                        inp = inp[diff:diff+rot_shape[1],0:rot_shape[0]]
        if self.random_crop:
            pad = (self.padding, self.padding, self.padding, self.padding)
            inp = torch.nn.functional.pad(inp, pad, "constant", 0)
            i = random.randint(0, 2*self.padding)
            j = random.randint(0, 2*self.padding)
            inp = inp[:,i:i+self.crop_shape[0],j:j+self.crop_shape[1]]
        
        # the number of channels should be 1 explicitly
        inp = torch.unsqueeze(inp, 0)
        
        return (inp, tg)
