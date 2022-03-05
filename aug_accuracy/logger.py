from torch.utils.tensorboard import SummaryWriter

import aug_accuracy.utils

class Logger:
    def __init__(self, log_path, model):
        self.writer = SummaryWriter(log_path)
        self.model = model
    
    def log_train(self, loss, epoch):
        print("{:05d} Training loss   = {}".format(epoch, loss))
        self.writer.add_scalar('Loss/train', loss, epoch)
        
    def log_validation(self, loss, epoch):
        print("{:05d} Validation loss = {}".format(epoch, loss))
        self.writer.add_scalar('Loss/validation', loss, epoch)
        
    def check_validation(self, val_dl, epoch):
        conf_mat, y_pred, y_true = self.model.test(val_dl)
        accuracy = aug_accuracy.utils.compute_accuracy(y_pred, y_true)
        self.writer.add_scalar('Accuracy/validation', accuracy, epoch)
