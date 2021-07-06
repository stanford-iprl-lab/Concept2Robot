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


class Master(nn.Module):
  def __init__(self, state_dim, action_dim, task_dim, max_action, params):
    super(Master, self).__init__()
    self.params = params
    self.model = models.resnet18(pretrained=True) 
    self.action_dim = action_dim
    self.max_action = max_action
    self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-2])  
    self.img_feat_block1 = nn.Sequential(
      nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
    )
    self.img_feat_dim = 256
    self.img_feat_block2 = nn.Linear(256 * 2 * 3, 256)
    self.img_feat_block3 = nn.Linear(256, 256)
    self.img_feat_block4 = nn.Linear(256, self.img_feat_dim)

    self.task_feat_block1 = nn.Linear(1024, 512)
    self.task_feat_block2 = nn.Linear(512, 256)
    self.task_feat_block3 = nn.Linear(256, 256)

    self.action_feat_block1 = nn.Linear(256+self.img_feat_dim, 256)
    self.action_feat_block5 = nn.Linear(256, 256)
    self.action_feat_block2 = nn.Linear(256, 256)
    self.action_feat_block3 = nn.Linear(256, 256)
    self.action_feat_block4 = nn.Linear(256, self.action_dim)

#####3 Force 
    # 1
    self.force_feat_block1 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256+self.img_feat_dim,out_channels=512,kernel_size=4,stride=1,bias=True),
      nn.ReLU(),
    )

    # 3
    self.force_feat_block2 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True),
      nn.ReLU(),
    )

    # 7
    self.force_feat_block3 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True),
      nn.ReLU(),
    )

    # 
    self.force_feat_block4 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=True),
      nn.ReLU(),
    )

    self.force_feat_block5 = nn.Sequential(
      nn.ConvTranspose1d(in_channels=256,out_channels=self.params.a_dim,kernel_size=3,stride=2,padding=1,bias=False),
    )

    set_init([self.img_feat_block2, self.img_feat_block3, self.img_feat_block4, self.task_feat_block1, self.task_feat_block2, self.task_feat_block3,\
      self.action_feat_block1, self.action_feat_block2, self.action_feat_block3, self.action_feat_block4,\
      self.action_feat_block5])

  def forward(self, state, task_vec):
    img_feat = self.feature_extractor(state) 
    img_feat = torch.tanh(self.img_feat_block1(img_feat))
    img_feat = img_feat.view(-1,256 * 2 * 3)
    img_feat = F.relu(self.img_feat_block2(img_feat))
    img_feat = F.relu(self.img_feat_block3(img_feat))
    img_feat = F.relu(torch.tanh(self.img_feat_block4(img_feat)))
 
    task_feat = F.relu(self.task_feat_block1(task_vec))
    task_feat = F.relu(self.task_feat_block2(task_feat))
    task_feat = F.relu(self.task_feat_block3(task_feat))

    task_feat = torch.cat([img_feat,task_feat],-1)
    
###################################################################
    action_feat = F.relu(self.action_feat_block1(task_feat))
    action_feat = F.relu(self.action_feat_block5(action_feat))
    action_feat = F.relu(self.action_feat_block2(action_feat))
    action_feat = F.relu(self.action_feat_block3(action_feat))
    goal = self.action_feat_block4(action_feat)

    force_feat_raw = task_feat
    force_feat = force_feat_raw.unsqueeze(2)
    force_feat = F.relu(self.force_feat_block1(force_feat))
    force_feat = F.relu(self.force_feat_block2(force_feat))
    force_feat = F.relu(self.force_feat_block3(force_feat))
    force_feat = F.relu(self.force_feat_block4(force_feat))
    force_feat = self.force_feat_block5(force_feat)
    _, n_dim, timesteps = force_feat.size()
    force = torch.transpose(force_feat, 1, 2)
    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((1, self.params.traj_timesteps, 1)) * float(
      self.params.traj_timesteps)
    weights = torch.FloatTensor(weights).to("cuda")
    force = weights * force
    return goal, force


class Master_F(nn.Module):
  def __init__(self, state_dim, action_dim, task_dim, max_action, params):
    super(Master_F, self).__init__()
    self.params = params
    self.model = models.resnet18(pretrained=True)
    self.action_dim = action_dim
    self.max_action = max_action
    self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-2])
    self.img_feat_block1 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True),
      nn.ReLU(),
      #      nn.BatchNorm2d(256),
    )
    self.img_feat_dim = 128
    self.img_feat_block2 = nn.Linear(256 * 2 * 3, 256)
    self.img_feat_block3 = nn.Linear(256 * self.params.stack_num, 128)
    self.img_feat_block4 = nn.Linear(128, self.img_feat_dim)

    self.task_feat_block1 = nn.Linear(1024, 512)
    self.task_feat_block2 = nn.Linear(512, 256)
    self.task_feat_block3 = nn.Linear(256, 256)

    self.action_feat_block1 = nn.Linear(256 + self.img_feat_dim, 256)
    self.action_feat_block2 = nn.Linear(256, 128)
    self.action_feat_block3 = nn.Linear(128, 64)
    self.action_feat_block4 = nn.Linear(64, self.action_dim)

    set_init(
      [self.img_feat_block2, self.img_feat_block3, self.img_feat_block4, self.task_feat_block1, self.task_feat_block2,
       self.task_feat_block3, \
       self.action_feat_block1, self.action_feat_block2, self.action_feat_block3, self.action_feat_block4])

  def forward(self, state, task_vec):
    img_feat = self.feature_extractor(state)
    img_feat = torch.tanh(self.img_feat_block1(img_feat))
    img_feat = img_feat.view(-1, 256 * 2 * 3)
    img_feat = F.relu(self.img_feat_block2(img_feat))
    img_feat = img_feat.view(-1, 256 * self.params.stack_num)
    img_feat = F.relu(self.img_feat_block3(img_feat))
    img_feat = F.relu(torch.tanh(self.img_feat_block4(img_feat)))

    task_feat = F.relu(self.task_feat_block1(task_vec))
    task_feat = F.relu(self.task_feat_block2(task_feat))
    task_feat = F.relu(self.task_feat_block3(task_feat))

    task_feat = torch.cat([img_feat, task_feat], -1)

    ###################################################################
    feedback = F.relu(self.action_feat_block1(task_feat))
    feedback = F.relu(self.action_feat_block2(feedback))
    feedback = F.relu(self.action_feat_block3(feedback))
    feedback = self.action_feat_block4(feedback)
    feedback = self.max_action * torch.tanh(feedback)
    return feedback
