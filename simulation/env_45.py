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

from env_43 import Engine43
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine45(Engine43):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine45,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        obj_pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        if obj_pos[2] > self.height + 0.05:
          return True
        else:
          return False
                          
