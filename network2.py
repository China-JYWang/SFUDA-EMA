import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=20),
                nn.BatchNorm1d(16),
                nn.PReLU(),

                nn.Conv1d(16, 32, kernel_size=5),
                nn.BatchNorm1d(32),
                nn.PReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),

                nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
                nn.BatchNorm1d(64),
                nn.PReLU(),

                nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.AdaptiveMaxPool1d(4))
        self.in_features = 128*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim=128*4, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.PReLU())
        self.bottleneck = nn.Linear(feature_dim, 256)
        self.bn = nn.BatchNorm1d(256, affine=True)
        self.em = nn.Embedding(2, 256)

    def forward(self, x, t, s=100,all_mask=False):
        x = self.bottleneck(x)
        x = self.bn(x)
        out=x
        if t == 0:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            out = out * self.mask
        if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            out = out * mask
        if all_mask:
            t0 = torch.LongTensor([0]).cuda()
            t1 = torch.LongTensor([1]).cuda()
            mask0 = nn.Sigmoid()(self.em(t0) * s)
            mask1 = nn.Sigmoid()(self.em(t1) * s)
            self.mask=mask0
            out0 = out * mask0
            out1 = out * mask1
        if all_mask:
            return (out0,out1), (self.mask,mask1)
        else:
            return out, self.mask

class feat_classifier(nn.Module):
    def __init__(self, class_num=4, bottleneck_dim=128, type="linear"):
        super(feat_classifier, self).__init__()
        # if type == "linear":
        #     self.fc = nn.Linear(bottleneck_dim, class_num)
        # else:
        #     self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        # self.fc.apply(init_weights)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(),

            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(),

            nn.Linear(64, class_num))


    def forward(self, x):
        x = self.fc(x)
        return x

