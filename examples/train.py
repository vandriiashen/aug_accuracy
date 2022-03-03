import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import aug_accuracy as util

def train(data_root, dataset_name, data_type, nn_type, run_num):
    data_folder = "{}/{}".format(data_root, dataset_name)
    train_input_glob = "{}/training/input/*.tiff".format(data_folder)
    train_target_glob = "{}/training/stats.csv".format(data_folder)
    val_input_glob = "{}/validation/input/*.tiff".format(data_folder)
    val_target_glob = "{}/validation/stats.csv".format(data_folder)
    save_path = "../network_state/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    Path(save_path).mkdir(exist_ok=True)
    Path(log_path).mkdir(exist_ok=True)
    save_path = Path(save_path)
    
    # Edit batch size depending on the GPU
    batch_size = 4
    
    train_ds = util.ImageDatasetTransformable(train_input_glob, train_target_glob, data_type,
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=True, horizontal_flip=True, rotate=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    val_ds = util.ImageDatasetTransformable(val_input_glob, val_target_glob, data_type,
             random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    c_in = 1
    if data_type == "playdoh":
        c_out = 2
    elif data_type == "avocado":
        c_out = 3
    else:
        raise InputError("Unknown sample type. Got {}".format(sample_type))
    
    model = util.NNmodel(c_in, c_out, nn_type)
    model.set_normalization(train_dl)

    best_validation_loss = np.inf
    prev_best_epoch = -1
    only_best_torch = True
    logger = util.Logger(log_path)
    
    # Edit the number of epochs depending on the convergence
    epochs = 1000
    
    for epoch in range(epochs):
        train_loss = model.train(train_dl)
        logger.log_train(train_loss, epoch)
        
        validation_loss = model.validate(val_dl)
        logger.log_validation(validation_loss, epoch)
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save(save_path / "{}.torch".format(epoch), epoch)
            if only_best_torch == True:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))
                prev_best_epoch = epoch

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=['playdoh', 'avocado'], help='Type of the dataset (playdoh or avocado)')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Optional, run number will be used as a seed if this argument is not provided')
    args = parser.parse_args()
    
    data_root = "/export/scratch2/vladysla/Data/Real/AugNN"
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    run_num = args.run
    if args.seed is not None:
        random_seed = args.seed
    else:
        random_seed = run_num
        
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    train(data_root, dataset_name, data_type, nn_type, run_num)
