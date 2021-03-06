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

def test(config, data_folder, epoch_file, data_type, nn_type):
    test_root = config[data_type]['test_folder']
    test_input_glob = "{}/input/*.tiff".format(test_root)
    test_target_glob = "{}/stats.csv".format(test_root)
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, config[data_type],
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    
    model = util.NNmodel(c_in, c_out, nn_type)
    model.load(epoch_file)
    
    conf_mat, y_pred, y_true = model.test(test_dl)
    print("Confusion matrix:")
    print(conf_mat)
    num_classes = config[data_type]['c_out']
    images_per_object = config[data_type]['img_per_obj']
    print("TP predictions split by objects:")
    print(util.utils.prediction_per_object(y_pred, y_true, num_classes, images_per_object))
    accuracy = util.utils.compute_accuracy(y_pred, y_true)
    print("Average accuracy = {:.3f}".format(accuracy))
    
    return accuracy

if __name__ == "__main__":
    config = util.utils.read_config('config.ini')
    data_keys = util.utils.get_available_data_types(config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=data_keys, help='Type of the dataset')
    args = parser.parse_args()
    
    data_root = "../network_state/"
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    
    # By default, use the epoch with the best validation score. If this flag is set to false, checkpoints could also be used.
    use_only_best = True
    
    data_root = Path(data_root)
    base_name = "{}_{}_r".format(dataset_name, nn_type)
    subfolders = [x for x in data_root.iterdir() if x.is_dir()]
    subfolders = filter(lambda x: x.name.startswith(base_name), subfolders)
    subfolders = sorted(subfolders)
    run_epochs = []
    acc_values = []
    for folder in tqdm(subfolders):
        saves = [x.stem for x in folder.glob("*.torch")]
        if use_only_best:
            only_final = filter(lambda x: not x.startswith('checkpoint'), saves)
            epochs = [int(x) for x in only_final]
            best_epoch = max(epochs)
            epoch_file = folder / "{}.torch".format(best_epoch)
        else:
            saves = sorted(saves)
            best_checkpoint = saves[-1]
            best_epoch = best_checkpoint.split('_')[-1]
            epoch_file = folder / "{}.torch".format(best_checkpoint)
        print("Network {}, epoch {}".format(folder.name, best_epoch))
        accuracy = test(config, folder, epoch_file, data_type, nn_type)
        run_epochs.append(best_epoch)
        acc_values.append(accuracy)
        
    pairs = []
    for i in range(len(run_epochs)):
        pairs.append("{},{:.3f}".format(run_epochs[i], acc_values[i]))
    print(",".join(pairs))
    print(",".join(pairs))
        

    
