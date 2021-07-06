import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np

import time
np.set_printoptions(precision=4,suppress=False)

import importlib
import imageio
import math
from tensorboardX import SummaryWriter
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMP_DIR = os.path.join(BASE_DIR,'../deepTraj')
PROJECT_DIR = os.path.join(BASE_DIR,'../')
sys.path.insert(0,DMP_DIR)
from ddmp import DDMP as DMP

sys.path.insert(0, "../simulation")

sys.path.insert(0, '../external/bullet3/build_cmake/examples/pybullet')

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from scipy.special import softmax

#####
from utils_config import load_args
from worker_model import Worker

if __name__ == "__main__":
    args = load_args()
    worker = Worker(args)
    print("args.comment",args.comment)
    if args.stage == 'test':
        worker.test(args.restore_episode, args.restore_path, args.restore_episode_goal, args.restore_path_goal)
    elif args.stage == "demonstration":
        worker.test(args.restore_episode, args.restore_path)
    elif args.stage == 'train':
        worker.train(args.restore_episode, args.restore_path, args.restore_episode_goal, args.restore_path_goal)
    elif args.stage == "imitation":
        worker.imitate()
    elif args.stage == "feedback_train":
        worker.feedback()
    elif args.stage == "imitation_test":
        worker.imitate_test(restore_episode=args.restore_episode,restore_path=args.restore_path)
    elif args.stage == "demonstration_feedback":
        worker.feedback_test(restore_feedback_episode=args.restore_feedback_episode, restore_feedback_path=args.restore_feedback_path)
    else:
        pass
