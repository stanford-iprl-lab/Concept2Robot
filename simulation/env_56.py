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

from env_55 import Engine55
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine56(Engine55):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine56,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        dist = np.linalg.norm(pos-self.pos)
        contact_info = self.p.getContactPoints (self.robotId, self.obj_id)
        if len(contact_info) > 0:
          self.contact = True
        if dist < 0.05 and self.contact and self.dmp.timestep >= self.dmp.timesteps:
          return True
        else:
          return False
