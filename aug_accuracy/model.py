# Scaling module is adapted from here
# https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/msd_model.py
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

def scaling_module(num_channels):
    c = nn.Conv2d(num_channels, num_channels, 1)
    c.bias.requires_grad = False
    c.weight.requires_grad = False
    scaling_module_set_scale(c, 1.0)
    scaling_module_set_bias(c, 0.0)

    return c

def scaling_module_set_scale(sm, s):
    c_out, c_in = sm.weight.shape[:2]
    assert c_out == c_in
    sm.weight.data.zero_()
    for i in range(c_out):
        sm.weight.data[i, i] = s

def scaling_module_set_bias(sm, bias):
    sm.bias.data.fill_(bias)
    
class NNmodel:
    def __init__(self, c_in, c_out, nn_type):
        self.c_in = c_in
        self.c_out = c_out
        self.scaling = scaling_module(c_in)
        
        self.set_classification_nn(c_in, c_out, nn_type)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def set_classification_nn(self, c_in, c_out, nn_type):
        if nn_type == "resnet50":
            self.classifier = models.resnet50(pretrained=False)
            self.classifier.conv1 = nn.Conv2d(c_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.classifier.fc = nn.Linear(in_features=2048, out_features=c_out, bias=True)
        else:
            raise InputError("Unknown sample type. Got {}".format(sample_type))
        
        self.net = nn.Sequential(self.scaling, self.classifier)
        self.net.cuda()
        self.init_optimizer(self.classifier)
        
    def init_optimizer(self, trainable_classifier):
        self.optimizer = torch.optim.SGD(trainable_classifier.parameters(), lr=0.001, momentum=0.9)
        
    def set_normalization(self, dl):
        mean = square = 0

        for (inp, tg) in dl:
            mean += inp.mean()
            square += inp.pow(2).mean()

        mean /= len(dl)
        square /= len(dl)
        
        std = np.sqrt(square - mean**2)

        scaling_module_set_scale(self.scaling, 1 / std)
        scaling_module_set_bias(self.scaling, -mean / std)
        
    def classify(self, inp):
        inp = inp.to('cuda')
        out = self.net(inp)
        return out
        
    def forward(self, inp, tg):
        inp = inp.to('cuda')
        tg = tg.to('cuda')
        out = self.net(inp)
        loss = self.criterion(out, tg)

        return loss
    
    def learn(self, inp, tg):
        loss = self.forward(inp, tg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def train(self, dl):
        avg_loss = 0
        for (inp, tg) in dl:
            avg_loss += self.learn(inp, tg).item()
        avg_loss /= len(dl)
        
        return avg_loss
    
    def validate(self, dl):
        avg_loss = 0
        for (inp, tg) in dl:
            avg_loss += self.forward(inp, tg).item()
        avg_loss /= len(dl)

        return avg_loss
    
    def save(self, path, epoch):
        state = {
            "epoch": int(epoch),
            "state_dict": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def test(self, dl):
        conf_mat = np.zeros((self.c_out, self.c_out))
        y_true = []
        y_pred = []
        
        for (inp, tg) in dl:
            out = self.classify(inp)
            out_max = torch.max(out, 1)
            gt = tg.item()
            prediction = out_max.indices.item()
            y_true.append(gt)
            y_pred.append(prediction)
            conf_mat[gt, prediction] += 1
            
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
            
        return (conf_mat, y_pred, y_true)
            

    def load(self, path):
        state = torch.load(path)
        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.net.cuda()
        epoch = state["epoch"]

        return epoch
