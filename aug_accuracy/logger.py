from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)
    
    def log_train(self, loss, epoch):
        print("{:05d} Training loss   = {}".format(epoch, loss))
        self.writer.add_scalar('Loss/train', loss, epoch)
        
    def log_validation(self, loss, epoch):
        print("{:05d} Validation loss = {}".format(epoch, loss))
        self.writer.add_scalar('Loss/validation', loss, epoch)
