#!/usr/bin/env python3
"""
    action101: put sth with sth
"""
import pybullet as p
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

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


################ Baseline Reward
import signal
import importlib
import torch
import torch.nn as nn

import sh
import re
import torch.nn.functional as F

np.set_printoptions(precision=4,suppress=True,linewidth=300)

class Engine24(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine24,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self.wid = worker_id
        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        #self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

        self.pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        self.orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))


    def init_obj(self):
        self.obj_scaling = 1.0
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/cylinders/c1/c1.urdf"),globalScaling=self.obj_scaling)
        self.obj2_scaling = 1.0
        self.obj2_pos = [0.29,0.1,0.31]
        self.obj2_ori = self.p.getQuaternionFromEuler([math.pi/8,0,0.])
        self.obj2_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bevels/b1/b1.urdf"),self.obj2_pos,self.obj2_ori,globalScaling=self.obj2_scaling,useFixedBase=True)

 
    def reset_obj(self):
        self.obj_position = [0.29, -0.1, 0.33]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        #p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    def init_grasp(self):

        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = self.p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        self.null_q = self.robot.getJointValue()#self.initial_pos

        self.obj_x, self.obj_y, self.obj_z = self.obj_position
        pos = [self.obj_x-0.03,self.obj_y-0.2,self.obj_z+0.0]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=20)

        self.init_height = self.p.getBasePositionAndOrientation(self.obj_id)[0][2]
        self.max_height = self.init_height
        self.reset_obj()

    def taskColliDet(self):
        colli = False
        for y in [0,1,2,3,4,5,6]:
          c = self.p.getContactPoints(bodyA=self.obj_id,bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
        return False

    def get_success (self,seg=None):
        pos1 = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        pos1 = np.array(pos1)
        if pos1[2] > self.max_height:
          self.max_height = pos1[2]
        if np.linalg.norm(self.init_height - pos1[2]) < 0.03 and self.max_height > self.init_height + 0.05:
          return True
        else:
          return False
