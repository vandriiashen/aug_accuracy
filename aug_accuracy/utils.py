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
    
