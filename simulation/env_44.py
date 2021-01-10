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
sys.path.append('./')

from env_41 import Engine41
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine44(Engine41):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine44,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        obj_pos = np.array(obj_pos)
        cam_pos = np.array([-0.09,0.04,0.89])
        dist_cam_obj = np.linalg.norm(cam_pos - obj_pos)
        #print("dist_cam_obj",dist_cam_obj,"dist",self.dist,"dist_cam_obj > self.dist + 0.1",dist_cam_obj > self.dist + 0.1)
        if dist_cam_obj < self.dist - 0.1:
          return True
        else:
          return False
                            
