import numpy as np
import torch
import random
import os
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import aug_accuracy as util
from aug_accuracy import InputError

def test(config, data_folder, epoch, data_type, nn_type):
    test_root = config[data_type]['test_folder']
    test_input_glob = "{}/input/*.tiff".format(test_root)
    test_target_glob = "{}/stats.csv".format(test_root)
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, config[data_type],
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    assert c_in == 1
    
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
    config = util.utils.read_config('config.ini')
    data_keys = util.utils.get_available_data_types(config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=data_keys, help='Type of the dataset')
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
    
    test(config, folder, best_epoch, data_type, nn_type)
