import numpy as np
import torch
import random
import os
import argparse
from pathlib import Path
from configparser import ConfigParser
from torch.utils.data import DataLoader

import aug_accuracy as util
from aug_accuracy import InputError

def load_data():
    input_glob = "../../../Data/Real/AugNN/p_t6/training/input/*.tiff"
    target_glob = "../../../Data/Real/AugNN/p_t6/training/stats.csv"
    
    batch_size = 4
    test_ds = util.ImageDatasetTransformable(input_glob, target_glob, "playdoh",
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=True, horizontal_flip=True, rotate=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    inp, tg = iter(test_dl).next()
    print(tg)
    plt.imshow(inp[0,:])
    plt.show()
    
def normalization():
    input_glob = "../../../Data/Real/AugNN/p_t6/training/input/*.tiff"
    target_glob = "../../../Data/Real/AugNN/p_t6/training/stats.csv"
    
    batch_size = 1
    ds = util.ImageDatasetTransformable(input_glob, target_glob, "playdoh",
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=True)
    dl = DataLoader(ds, batch_size, shuffle=False)
    
    mean = square = 0
    for (inp, tg) in dl:
        mean += inp.mean()
        square += inp.pow(2).mean()
    mean /= len(dl)
    square /= len(dl)
    std = np.sqrt(square - mean**2)
    print(mean, square, std)
    
    model = util.NNmodel(1, 2, "resnet50")
    model.set_normalization(dl)
    print(model.scaling)
    print(model.scaling.weight.data)
    print(model.scaling.bias.data)

    mean = square = 0
    for (inp, tg) in dl:
        normalized_inp = model.scaling(inp)
        mean += normalized_inp.mean()
        square += normalized_inp.pow(2).mean()
    mean /= len(dl)
    square /= len(dl)
    std = np.sqrt(square - mean**2)
    print(mean, square, std)
    
def train_test():
    model_name = "p_t6"
    nn_type = "resnet50"
    run_num = 1
    train_input_glob = "../../../Data/Real/AugNN/p_t6/training/input/*.tiff"
    train_target_glob = "../../../Data/Real/AugNN/p_t6/training/stats.csv"
    val_input_glob = "../../../Data/Real/AugNN/p_t6/training/input/*.tiff"
    val_target_glob = "../../../Data/Real/AugNN/p_t6/training/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    save_path.mkdir(exist_ok=True)
    
    parser = util.utils.read_config('config.ini')
    dtype_conf = parser['playdoh']
    
    batch_size = 4
    train_ds = util.ImageDatasetTransformable(train_input_glob, train_target_glob, dtype_conf,
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=True, horizontal_flip=True, rotate=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    val_ds = util.ImageDatasetTransformable(val_input_glob, val_target_glob, dtype_conf,
             random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    inp, tg = iter(train_dl).next()
    print(inp.type())
    
    model = util.NNmodel(1, 2, nn_type)
    model.set_normalization(train_dl)

    best_validation_loss = np.inf
    prev_best_epoch = -1
    only_best_torch = True

    epochs = 5
    for epoch in range(epochs):
        train_loss = model.train(train_dl)
        print("{:05d} Training loss   = {}".format(epoch, train_loss))
        validation_loss = model.validate(val_dl)
        print("{:05d} Validation loss = {}".format(epoch, validation_loss))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save(save_path / "{}.torch".format(epoch), epoch, batch_size)
            if only_best_torch == True:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))
                prev_best_epoch = epoch
                
def apply_single_test():
    model_name = "p_t6"
    nn_type = "resnet50"
    run_num = 1
    epoch = 3
    test_input_glob = "../../../Data/Real/AugNN/test_playdoh3/input/*.tiff"
    test_target_glob = "../../../Data/Real/AugNN/test_playdoh3/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, "playdoh",
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    model = util.NNmodel(1, 2, nn_type)
    model.load(save_path / "{}.torch".format(epoch))
    
    inp, tg = iter(test_dl).next()
    out = model.classify(inp)
    print(out)
    print(tg)
    out_max = torch.max(out, 1)
    print(out_max.indices.item())
    
def apply_data_test():
    model_name = "p_t6"
    nn_type = "resnet50"
    run_num = 1
    epoch = 2
    test_input_glob = "../../../Data/Real/AugNN/test_playdoh3/input/*.tiff"
    test_target_glob = "../../../Data/Real/AugNN/test_playdoh3/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, "playdoh",
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    model = util.NNmodel(1, 2, nn_type)
    model.load(save_path / "{}.torch".format(epoch))
    
    conf_mat, y_pred, y_true = model.test(test_dl)
    print(conf_mat)
    print(util.utils.compute_accuracy(y_pred, y_true))
    print(util.utils.split_prediction(y_pred, y_true, 45))
    
def cli_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=['playdoh', 'avocado', 'avocado_binary', 'avocado_multi'], help='Type of the dataset (playdoh or avocado)')
    parser.add_argument('--run', type=int, required=True, help='Run number, also used as a random seed')
    parser.add_argument('--seed', type=int, required=False, help='Random seed. Optional, run number will be used as a seed if this argument is not provided')
    args = parser.parse_args()
    
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    run_num = args.run
    if args.seed is not None:
        random_seed = args.seed
    else:
        random_seed = run_num
    
    data_root = "/export/scratch2/vladysla/Data/Real/AugNN"
    data_root = "{}/{}".format(data_root, dataset_name)
    train_input_glob = "{}/training/input/*.tiff".format(data_root)
    train_target_glob = "{}/training/stats.csv".format(data_root)
    val_input_glob = "{}/validation/input/*.tiff".format(data_root)
    val_target_glob = "{}/validation/stats.csv".format(data_root)
    print(train_input_glob)
    
    save_path = "../network_state/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    print(save_path)
    print(log_path)
    
    model = util.NNmodel(1, 2, nn_type)
    print(model.net[1])
        
    print("Random seed = ", random_seed)
    
def activation_test():
    model_name = "av_t2"
    nn_type = "resnet50"
    run_num = 2
    epoch = 419
    test_input_glob = "../../../Data/Real/AugNN/test_avocado3/input/*.tiff"
    test_target_glob = "../../../Data/Real/AugNN/test_avocado3/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    batch_size = 1
    test_ds = util.ImageDatasetTransformable(test_input_glob, test_target_glob, "avocado",
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
    model = util.NNmodel(1, 3, "resnet50")
    model.load(save_path / "{}.torch".format(epoch))
    #print(model.net)
    
    cam = util.ActivationMap(model)
    
    inp, tg = iter(test_dl).next()
    tg = tg.item()
    out = model.classify(inp)
    out_class = torch.max(out, 1).indices.item()
    cam.visualize(inp, tg, out_class, "cam.png")
    
def multi_test():
    model_name = "av_m1"
    nn_type = "resnet50"
    run_num = 1
    train_input_glob = "../../../Data/Real/AugNN/av_m1/training/input/*.tiff"
    train_target_glob = "../../../Data/Real/AugNN/av_m1/training/stats.csv"
    val_input_glob = "../../../Data/Real/AugNN/av_m1/training/input/*.tiff"
    val_target_glob = "../../../Data/Real/AugNN/av_m1/training/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    save_path.mkdir(exist_ok=True)
    
    batch_size = 1
    train_ds = util.ImageDatasetTransformable(train_input_glob, train_target_glob, "avocado_multi",
               random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=True, horizontal_flip=True, rotate=True)
    train_dl = DataLoader(train_ds, batch_size, shuffle=False)
    val_ds = util.ImageDatasetTransformable(val_input_glob, val_target_glob, "avocado_multi",
             random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    inp, tg = iter(train_dl).next()
    print(inp.type())
    
    model = util.NNmodel(30, 2, nn_type)
    model.set_normalization(train_dl)

    best_validation_loss = np.inf
    prev_best_epoch = -1
    only_best_torch = True

    epochs = 5
    for epoch in range(epochs):
        train_loss = model.train(train_dl)
        print("{:05d} Training loss   = {}".format(epoch, train_loss))
        validation_loss = model.validate(val_dl)
        print("{:05d} Validation loss = {}".format(epoch, validation_loss))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save(save_path / "{}.torch".format(epoch), epoch, batch_size)
            if only_best_torch == True:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))
                prev_best_epoch = epoch
                
def config_test():
    parser = ConfigParser()
    parser.read("config.ini")
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    config_sections = list(config.keys())
    config_sections.remove('General')
    print(config_sections)
    print(config['playdoh'])

def val_test():
    model_name = "avo_t3"
    nn_type = "resnet50"
    run_num = 2
    epoch = 8
    val_input_glob = "../../../Data/Real/AugNN/avo_t3/validation/input/*.tiff"
    val_target_glob = "../../../Data/Real/AugNN/avo_t3/validation/stats.csv"
    #test_input_glob = "../../../Data/Real/AugNN/test_avo_inp1_3class/input/*.tiff"
    #test_target_glob = "../../../Data/Real/AugNN/test_avo_inp1_3class/stats.csv"
    save_path = Path("../network_state/{}_{}_r{}/".format(model_name, nn_type, run_num))
    
    config = util.utils.read_config('config.ini')
    
    batch_size = 1
    val_ds = util.ImageDatasetTransformable(val_input_glob, val_target_glob, config["avocado"],
              random_crop=False, padding=20, crop_shape=(380,478), vertical_flip=False, horizontal_flip=False, rotate=False)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    
    model = util.NNmodel(1, 3, nn_type)
    model.load(save_path / "{}.torch".format(epoch))
    
    print(model.validate(val_dl))
        
if __name__ == "__main__":
    random_seed = 2
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    #load_data()
    #normalization()
    #train_test()
    #apply_single_test()
    #apply_data_test()
    #cli_test()
    #activation_test()
    #multi_test()
    #config_test()
    val_test()
