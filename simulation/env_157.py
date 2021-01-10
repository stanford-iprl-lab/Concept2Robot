#!/usr/bin/env python3

import time
import math
from datetime import datetime
from time import sleep
import numpy as np
import random
import cv2
import os
import argparse
import torch

import sys
sys.path.append('./Eval')
sys.path.append('./')

from env_156 import Engine156
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine157(Engine156):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine157,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=True)
        self.opti = opti

    def get_success(self,seg=None):
        box_AABB = self.p.getAABB(self.obj_id)
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        if obj_pos[0] > box_AABB[0][0] and obj_pos[0] < box_AABB[1][0] and obj_pos[1] > box_AABB[0][1] and obj_pos[1] < box_AABB[1][1]:
          return False
        else:
          return True

