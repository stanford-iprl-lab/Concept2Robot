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
sys.path.append('./')

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


class Engine94(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine94,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self.wid = worker_id
        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

        self.pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        self.orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.pos = None

    def init_obj(self):
        self.obj_file = os.path.join(self.resources_dir, "urdf/objmodels/nut.urdf")
        self.obj_position = [0.4, -0.15, 0.34]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi / 2, -math.pi / 2, 0])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,
                                      baseOrientation=self.obj_orientation,
                                      globalScaling=self.obj_scaling)  # ,physicsClientId=self.physical_id)
        self.p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.3, 0.3, 0.9, 1])

    def reset_obj(self):
        obj_x = 0.35
        obj_y = -0.05
        transl = np.random.uniform(-0.075,0.075,size=(2,))
        self.obj_position = [obj_x + transl[0], obj_y + transl[1], 0.31]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    def init_grasp(self):
        self.robot.gripperControl(0)
        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[2] += 0.1
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=150)

        cur_joint = self.robot.getJointValue()
        cur_pos = np.array(self.obj_position)#self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        cur_pos[1] += -0.2
        cur_pos[0] += -0.08
        cur_pos[2] += 0.1
        for i in range(109):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=150)

        pos = self.robot.getEndEffectorPos()
        pos[2] -= 0.02
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=150)
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        self.pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])

    def get_success(self,seg=None):
        pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        if pos[1] > self.pos[1] + 0.1:
          return True
        else:
          return False

