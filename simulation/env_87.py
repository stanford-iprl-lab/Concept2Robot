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

class Engine87(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine87,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self.pos = None  
        expert_traj_dir = os.path.join(self.robot_recordings_dir,"87-3")
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
        self.obj_position = [0.4, -0.03, 0.34]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

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


    def init_grasp(self):
        self.reset_obj()
        #self.p.changeDynamics (self.obj_id, -1, mass=0.0)
        self.robot.setJointValue(self.data_q[0],gripper=150)

        self.null_q = self.data_q[0]
        orn = self.orn_traj[0]
        #self.p.changeDynamics (self.obj_id, -1, mass=10.0)
        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] += 0.05
        print("before initialiaztion")
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=150)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]

    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj_pos[1] > self.pos[1] + 0.1:
          return True
        else:
          return False


