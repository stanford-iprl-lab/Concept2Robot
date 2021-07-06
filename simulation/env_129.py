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

import matplotlib.pyplot as plt
import sys
sys.path.append('./Eval')
sys.path.append('./')

from env import Engine
from robot_cup_2 import Robot
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine129(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine129,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=Robot)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 10000.0
        self.robot.armMaxForce = 20000.0
        self.robot.jd = [0.01] * 14
        self.p = p_id
        self.p.setGravity(0,0,-900.81)
        self.p.setTimeStep(1 / 30.0)

        expert_traj_dir = os.path.join(self.robot_recordings_dir,"87-3")
        self.data_q = np.load(os.path.join(expert_traj_dir,'q.npy'))
        self.data_dq = np.load(os.path.join(expert_traj_dir,'dq.npy'))


    def reset_new(self):

        self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
      
        return self.get_observation()

    def init_obj(self):
        pass
 

    def reset_obj(self,pos=np.array([0.3,0.1,0.5])):
        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)
        #self.p.resetBasePositionAndOrientation(self.obj_id,pos,self.obj_orientation)
        #self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.949,0.878,0.0392,1.0])
    
    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        self.robot.setJointValue(self.data_q[40],gripper=250)
        pos = self.robot.getEndEffectorPos()
        orn = self.robot.getEndEffectorOrn()
        pos[2] += 0.2
#        pos[0] = self.box_position[0]
#        pos[1] = self.box_position[1] + 0.08
        self.null_q = self.data_q[0]
  
        for i in range(50):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=0)
        
        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] -= 0.04
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=0)

        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_observation_bo(self,segFlag=False):
        # get observation
        img_info = self.p.getCameraImage (width=self.w,
                                     height=self.h,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     shadow=-1,
                                     flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                     renderer=self.p.ER_TINY_RENDERER)
        img = img_info[2][:, :, :3]

        self.observation = img
        if segFlag:
            seg = img_info[4]
            return img, seg
        else:
            return img

    def get_success (self,seg=None):
        img = self.get_observation_bo()
        pixels = np.sum( (img[:,:,1] > 0.85) * (img[:,:,0] < 0.05) )
        if pixels > 100:
          return True
        else:
          return False
