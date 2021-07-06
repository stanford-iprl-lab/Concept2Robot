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

try:
    from .env import Engine
    from .utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
except Exception:
    from env import Engine
    from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

#################################
import signal
import importlib
import torch
import torch.nn as nn

import sh
import re
import torch.nn.functional as F

np.set_printoptions(precision=4,suppress=True,linewidth=300)

class Engine96(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine96,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
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
        self.p.setGravity(0, 0, -9.81)

        expert_traj_dir = os.path.join(self.robot_recordings_dir,"87-0")
        self.data_q = np.load(os.path.join(expert_traj_dir,'q.npy'))
        self.data_dq = np.load(os.path.join(expert_traj_dir,'dq.npy'))

        pos_list = []
        orn_list = []
        for i in range(len(self.data_q)):
          self.robot.setJointValue(self.data_q[i],gripper=220)
          pos = self.robot.getEndEffectorPos()
          pos_list.append(pos)
          orn = self.robot.getEndEffectorOrn()
          orn_list.append(orn)
        self.pos_traj = pos_list
        self.orn_traj = orn_list

        self.fix_orn = orn_list

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.35, 0.1, 0.31]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)


        self.box_file = os.path.join (self.resources_dir, "urdf/obj_libs/cubes/c3/c3.urdf")
        self.box_position = [0.48, -0.1, 0.27]
        self.box_scaling = 1.5
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0, 0.3, 0.6, 1])

    def reset_obj(self):
        box_x = 0.48
        box_y = -0.1
        transl = np.random.uniform(-0.05,0.05,size=(2,))
        self.box_position = [box_x + transl[0],box_y + transl[1], 0.256]
        self.p.resetBasePositionAndOrientation (self.box_id, self.box_position, self.box_orientation)
        self.p.changeVisualShape (self.box_id, -1, rgbaColor=[1.0, 0.0, 0.0, 1])

        obj_x = 0.35
        obj_y = 0.1
        transl_obj = np.random.uniform(-0.05,0.05,size=(2,))
        self.obj_position = [obj_x + transl_obj[0],obj_y + transl_obj[1], 0.33]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        for i in range(5):
          self.p.stepSimulation()


    def init_grasp(self):
        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
        self.robot.gripperControl(0)

        self.null_q = self.data_q[0]
        orn = self.orn_traj[0]

        self.fix_orn = np.array(orn)
        self.fix_orn = np.expand_dims(orn,axis=0)


        pos = [self.obj_position[0]-0.03, self.obj_position[1]+0.2, self.obj_position[2] + 0.18]
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        pos = [self.obj_position[0]-0.03, self.obj_position[1]+0.15, self.obj_position[2] + 0.075]
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
 
    def get_success(self,seg=None):
        box_AABB = self.p.getAABB(self.box_id)
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        closet_info = self.p.getContactPoints (self.box_id, self.obj_id)
        offset = 0.03
        if len(closet_info) > 0 and obj_pos[0] > box_AABB[0][0] + offset and obj_pos[0] < box_AABB[1][0] - offset and obj_pos[1] > box_AABB[0][1] + offset and obj_pos[1] < box_AABB[1][1] - offset:
          return True
        else:
          return False


