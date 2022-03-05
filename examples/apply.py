import numpy as np
import torch
import random
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import aug_accuracy as util

def test(data_folder, epoch, data_type, nn_type):
    test_input_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_playdoh3/input/*.tiff"
    test_target_glob = "/export/scratch2/vladysla/Data/Real/AugNN/test_playdoh3/stats.csv"
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, data_type,
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    c_in = 1
    if data_type == "playdoh":
        c_out = 2
    elif data_type == "avocado":
        c_out = 3
    else:
        raise InputError("Unknown sample type. Got {}".format(sample_type))
    
    model = util.NNmodel(c_in, c_out, nn_type)
    model.load(data_folder / "{}.torch".format(epoch))
    
    conf_mat, y_pred, y_true = model.test(test_dl)
    print(conf_mat)
    print(util.utils.split_prediction(y_pred, y_true, 45))
    accuracy = util.utils.compute_accuracy(y_pred, y_true)
    
    return accuracy

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=['playdoh', 'avocado'], help='Type of the dataset (playdoh or avocado)')
    args = parser.parse_args()
    
    data_root = "../network_state/"
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    
    data_root = Path(data_root)
    base_name = "{}_{}_r".format(dataset_name, nn_type)
    subfolders = [x for x in data_root.iterdir() if x.is_dir()]
    subfolders = filter(lambda x: x.name.startswith(base_name), subfolders)
    run_epochs = []
    acc_values = []
    for folder in subfolders:
        epochs = [int(x.stem) for x in folder.glob("*.torch")]
        best_epoch = max(epochs)
        print("Network {}, epoch {}".format(folder.name, best_epoch))
        accuracy = test(folder, best_epoch, data_type, nn_type)
        run_epochs.append(best_epoch)
        acc_values.append(accuracy)
        
    pairs = []
    for i in range(len(run_epochs)):
        pairs.append("{},{:.3f}".format(run_epochs[i], acc_values[i]))
    print(",".join(pairs))
    print(",".join(pairs))
        

    
