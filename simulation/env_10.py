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

from env_8 import Engine8
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine10(Engine8):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine10,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])][:2]
        box_center = np.array(box_center)
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])][:2]
        obj_center = np.array(obj_center)
        dist = np.linalg.norm(box_center - obj_center)
        box_corner = np.array([box[0][0], box[0][1]])
        box_radius = np.linalg.norm(box_corner - box_center)
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        obj_box_closet_info = self.p.getContactPoints (self.box_id, self.obj_id, -1, -1)
        box = self.p.getAABB (self.box_id)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])][:2]
        box_center = np.array(box_center)
        obj = self.p.getAABB (self.obj_id)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])][:2]
        obj_center = np.array(obj_center)
        dist = np.linalg.norm(box_center - obj_center)
        closet_info = self.p.getContactPoints (self.box_id, self.obj_id, linkIndexA=-1,linkIndexB=-1)
        box_corner = np.array([box[0][0], box[0][1]])
        box_radius = np.linalg.norm(box_corner - box_center)
        obj_corner = np.array([obj[0][0], obj[0][1]])
        obj_radius = np.linalg.norm(obj_corner - obj_center)
        box_h_max = self.p.getAABB(self.box_id,1)[1][2]
        print("box_h_max",box_h_max)
        print("close",len(closet_info),"dist",dist , "<", box_radius - obj_radius * 0.5, "obj_raidus",obj_radius, obj[1][2], "<", box_h_max + obj_radius, obj[0][2], '<', box_h_max - obj_radius * 0.25)
        if len (left_closet_info)==0 and len (right_closet_info)==0 and dist < box_radius - obj_radius and ( len(closet_info) > 0 or obj[1][2]  < box_h_max + obj_radius or obj[0][2] < box_h_max - obj_radius):
          return True
        else:
          return False
