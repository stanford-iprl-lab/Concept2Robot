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

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine89(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine89,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.06, 0.35]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.3])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        self.obj2_file = os.path.join(self.urdf_dir,"obj_libs/bottles/b6/b6.urdf")
        self.obj2_position = [0.3637 + 0.06, 0.06, 0.28]
        self.obj2_scaling = 3
        self.obj2_orientation = self.p.getQuaternionFromEuler([0,0,0])
        self.obj2_id = self.p.loadURDF(fileName=self.obj2_file, basePosition=self.obj2_position,baseOrientation=self.obj2_orientation,
                                     globalScaling=self.obj2_scaling,useFixedBase=True)#,physicsClientId=self.physical_id


        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[0.949, 0.07, 0.0392,1.0])   

        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=20.0)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.1)


    def reset_obj(self):
       self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)


    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):

        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(150,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[1] += 0.1
        pos[0] += 0.0
        pos[2] += 0.05
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=150)
#        time.sleep(3)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]


    def get_success(self,seg=None):
        box_AABB = self.p.getAABB(self.obj2_id) 
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        if obj_pos[0] > box_AABB[0][0] and obj_pos[0] < box_AABB[1][0] and obj_pos[1] > box_AABB[0][1] and obj_pos[1] < box_AABB[1][1]: 
          return False
        else:
          return True
