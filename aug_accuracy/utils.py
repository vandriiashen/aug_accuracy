from configparser import ConfigParser
import numpy as np
from sklearn.metrics import accuracy_score

def split_prediction(y_pred, y_true, obj_proj):
    """Compare class predictions for images corresponding to different objects
    """
    total_num = y_pred.shape[0]
    num_obj = total_num // obj_proj
    assert total_num % obj_proj == 0
    
    class_seq = []
    for i in range(num_obj):
        counter = 0
        for j in range(obj_proj):
            if y_pred[i*obj_proj+j] == y_true[i*obj_proj+j]:
                counter += 1
        class_seq.append(counter)
    
    return "[" + ",".join(str(el) for el in class_seq) + "]"

def compute_accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)
    
def read_config(fname):
    parser = ConfigParser()
    parser.read(fname)
    config = {s:dict(parser.items(s)) for s in parser.sections()}
    
    data_dict_keys = list(config.keys())
    data_dict_keys.remove('General')
    
    config['General']['batch_size'] = int(parser['General'].get('batch_size', -1))
    config['General']['max_epochs'] = int(parser['General'].get('max_epochs', -1))
    config['General']['use_deterministic'] = parser['General'].getboolean('use_deterministic', False)
    for key in data_dict_keys:
        config[key]['c_in'] = int(parser[key].get('c_in', -1))
        config[key]['c_out'] = int(parser[key].get('c_out', -1))
    
    return config

def get_available_data_types(config):
    data_dict_keys = list(config.keys())
    data_dict_keys.remove('General')
    return data_dict_keys
