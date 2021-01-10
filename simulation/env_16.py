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

from env_8 import Engine8
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

class Engine16(Engine8):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine16,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def step_dmp(self,action,f_w,coupling,reset,test=False):
        if reset:
          action = action.squeeze()
          self.start_pos = self.robot.getEndEffectorPos()
          self.start_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
          self.start_gripper_pos = self.robot.getGripperPos()
          self.start_status = np.array([self.start_pos[0],self.start_pos[1],self.start_pos[2],self.start_orn[0],self.start_orn[1],self.start_orn[2],0.0]).reshape((-1,))
          self.dmp.set_start(np.array(self.start_status)[:self.dmp.n_dmps])
          dmp_end_pos = [x+y for x,y in zip(self.start_status,action)]
          self.dmp.set_goal(dmp_end_pos)
          if f_w is not None:
            self.dmp.set_force(f_w)
          self.dmp.reset_state()
          #self.traj = self.dmp.gen_traj()
          self.actual_traj = []
          p1 = self.start_pos 
          p1 = np.array(p1)
          self.dmp.timestep = 0
          small_observation = self.step_within_dmp (coupling)
          lenT = len(self.dmp.force[:,0])
        else:
          small_observation = self.step_within_dmp(coupling)
        seg = None
        observation_next, seg = self.get_observation(segFlag=True)
        reward = 0
        done = False
        suc = False
        suc_info = self.get_success()
        if self.dmp.timestep >= self.dmp.timesteps:
          print("seg",seg)
          reward = self.get_reward(seg)
          done = True
          self.success_flag = suc_info
        else:
          if np.sum(seg == 167772162) < 1:
            done = True
            self.success_flag = False
        return observation_next, reward, done, self.success_flag

    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]

        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info) > 0 and len (right_closet_info) > 0:
          return True
        else:
          return False
