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
from math import sin,cos,acos
import matplotlib.pyplot as plt

import sys
sys.path.append('./Eval')
sys.path.append('./')

from env_104 import Engine104
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

def angleaxis2quaternion(angleaxis):
  angle = np.linalg.norm(angleaxis)
  axis = angleaxis / (angle + 0.00001)
  q0 = cos(angle/2)
  qx,qy,qz = axis * sin(angle/2)
  return np.array([qx,qy,qz,q0])

def quaternion2angleaxis(quater):
  angle = 2 * acos(quater[3])
  axis = quater[:3]/(sin(angle/2)+0.00001)
  angleaxis = axis * angle
  return np.array(angleaxis)

class Engine107(Engine104):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine107,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])][:2]
        box_center = np.array(box_center)
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])][:2]
        obj_center = np.array(obj_center)
        dist = np.linalg.norm(box_center - obj_center)
        closet_info = self.p.getContactPoints (self.box_id, self.obj_id, linkIndexA=-1,linkIndexB=-1)
        box_corner = np.array([box[0][0], box[0][1]])
        box_radius = np.linalg.norm(box_corner - box_center)
        obj_corner = np.array([obj[0][0], obj[0][1]])
        obj_radius = np.linalg.norm(obj_corner - obj_center)
        print("dist",dist,"close",len(closet_info),"radius",box_radius - obj_radius,  obj[1][2], box[1][2])
        if dist > box_radius + obj_radius and len(closet_info) > 0 and dist < box_radius + obj_radius * 4:
          return True
        else:
          return False
