import numpy as np
import torch
import random
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import aug_accuracy as util
from aug_accuracy import InputError

def train(config, dataset_name, data_type, nn_type, run_num):
    data_root = config['General']['data_root']
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
    batch_size = config['General']['batch_size']
    
    train_ds = util.ImageDatasetTransformable(train_input_glob, train_target_glob, config[data_type],
               random_crop=True, padding=20, crop_shape=(380,478), vertical_flip=True, horizontal_flip=True, rotate=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_ds = util.ImageDatasetTransformable(val_input_glob, val_target_glob, config[data_type],
             random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    train_ds.check_class_frequency()
    val_ds.check_class_frequency()
    
    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    
    model = util.NNmodel(c_in, c_out, nn_type)
    model.set_normalization(train_dl)

    best_validation_loss = np.inf
    prev_best_epoch = -1
    only_best_torch = True
    logger = util.Logger(log_path, model)
    
    # Edit the number of epochs depending on the convergence
    epochs = config['General']['max_epochs']
    
    for epoch in tqdm(range(epochs)):
        train_loss = model.train(train_dl)
        logger.log_train(train_loss, epoch)
        
        validation_loss = model.validate(val_dl)
        logger.log_validation(validation_loss, epoch)
        
        if epoch % 50 == 0:
            logger.check_validation(val_dl, epoch)
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save(save_path / "{}.torch".format(epoch), epoch, batch_size)
            if only_best_torch == True:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))
                prev_best_epoch = epoch
                
        if epoch % 100 == 0:
            model.save(save_path / "checkpoint_{}.torch".format(epoch), epoch, batch_size)

if __name__ == "__main__":
    config = util.utils.read_config('config.ini')
    data_keys = util.utils.get_available_data_types(config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=data_keys, help='Type of the dataset')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Optional, run number will be used as a seed if this argument is not provided')
    args = parser.parse_args()

    data_root = config['General']['data_root']
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
    torch.manual_seed(random_seed)
    if config['General']['use_deterministic']:
        print("Use detereterministic algorithms")
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        print("Use faster algorithms")
    
    train(config, dataset_name, data_type, nn_type, run_num)
