import numpy as np
import torch
import random
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import aug_accuracy as util
from aug_accuracy import InputError

def test(data_folder, epoch, data_type, nn_type):
    if data_type == "playdoh":
        test_input_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_playdoh3/input/*.tiff"
        test_target_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_playdoh3/stats.csv"
    elif data_type == "avocado":
        test_input_glob = "/export/scratch2/vladysla/Data/Real/AugNN/av_t4/training/input/*.tiff"
        test_target_glob = "/export/scratch2/vladysla/Data/Real/AugNN/av_t4/training/stats.csv"
    elif data_type == "avocado_binary":
        test_input_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_avocado3/input/*.tiff"
        test_target_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_avocado3/stats.csv"
    else:
        raise InputError("Unknown sample type. Got {}".format(sample_type))
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, data_type,
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    c_in = 1
    if data_type == "playdoh":
        c_out = 2
    elif data_type == "avocado":
        c_out = 3
    elif data_type == "avocado_binary":
        c_out = 2
    else:
        raise InputError("Unknown sample type. Got {}".format(sample_type))
    
    model = util.NNmodel(c_in, c_out, nn_type)
    model.load(data_folder / "{}.torch".format(epoch))
    cam = util.ActivationMap(model)
    
    img_folder = Path("../res")
    img_folder.mkdir(exist_ok=True)
    i = 0
    for (inp, tg) in test_dl:
        tg = tg.item()
        out = model.classify(inp)
        out_class = torch.max(out, 1).indices.item()
        cam.visualize(inp, tg, out_class, img_folder / "{}.png".format(i))
        i += 1
    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=['playdoh', 'avocado', 'avocado_binary'], help='Type of the dataset (playdoh or avocado)')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    args = parser.parse_args()
    
    data_root = "../network_state"
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    run_num = args.run
    
    folder = Path("{}/{}_{}_r{}/".format(data_root, dataset_name, nn_type, run_num))
    epochs = [int(x.stem) for x in folder.glob("*.torch")]
    best_epoch = max(epochs)
    
    test(folder, best_epoch, data_type, nn_type)
