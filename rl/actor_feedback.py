import copy
import numpy as np
import os
import sys
import time
np.set_printoptions(precision=4,suppress=False)

import importlib
import glob
import imageio
import math
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

def set_init(layers):
  for layer in layers:
    nn.init.normal_(layer.weight, mean=0., std=0.05)
    nn.init.constant_(layer.bias, 0.)

class Actor_F(nn.Module):
  def __init__(self, state_dim, action_dim, task_dim, max_action, params):
    super(Actor_F, self).__init__()
    self.params = params
    self.model = models.resnet18(pretrained=False) 
    self.action_dim = action_dim
    self.max_action = max_action
    self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-2])  
    self.img_feat_block1 = nn.Sequential(
      nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(256),
    )
    self.img_feat_block2 = nn.Linear(256 * 2 * 3, 256)

    self.task_feat_block1 = nn.Linear(1024, 512)
    self.task_feat_block2 = nn.Linear(512, 256)
    self.task_feat_block3 = nn.Linear(256, 128)

    self.coupling_feat_block1 = nn.Linear(256 + 128, 256)
    self.coupling_feat_block2 = nn.Linear(256, 128)
    self.coupling_feat_block3 = nn.Linear(128, 64)
    self.coupling_feat_block4 = nn.Linear(64, self.action_dim)

    set_init([self.img_feat_block2, self.task_feat_block1, self.task_feat_block2, self.task_feat_block3,\
      self.coupling_feat_block1,self.coupling_feat_block2,self.coupling_feat_block3,self.coupling_feat_block4])

  def forward(self, state, task_vec):
    img_feat = self.feature_extractor(state) 
    img_feat = self.img_feat_block1(img_feat)
    img_feat = img_feat.view(-1,256 * 2 * 3)
    img_feat = self.img_feat_block2(img_feat) 

    task_feat = F.relu(self.task_feat_block1(task_vec))
    task_feat = F.relu(self.task_feat_block2(task_feat))
    task_feat = F.relu(self.task_feat_block3(task_feat))

    coupling_feat_raw = torch.cat([img_feat,task_feat],-1)

    coupling_feat = F.relu(self.coupling_feat_block1(coupling_feat_raw))
    coupling_feat = F.relu(self.coupling_feat_block2(coupling_feat))
    coupling_feat = F.relu(self.coupling_feat_block3(coupling_feat))
    coupling_feat = F.relu(self.coupling_feat_block4(coupling_feat))
    coupling = self.max_action * torch.tanh(coupling_feat)
    return coupling

