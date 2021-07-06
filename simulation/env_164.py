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
from PIL import Image

import sys
sys.path.append('./Eval')
sys.path.append('./')

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from env import Engine
from robot_cup import Robot
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine164(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine164,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=Robot)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 10000.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

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
#        pass
        self.p.addUserDebugLine([0,0,0],[0.2,0,0],[5,0,0],parentObjectUniqueId=self.robotId, parentLinkIndex=self.robot.base_index)
        self.p.addUserDebugLine([0,0,0],[0,0.2,0],[0,5,0],parentObjectUniqueId=self.robotId, parentLinkIndex=self.robot.base_index)
        self.p.addUserDebugLine([0,0,0],[0,0,0.2],[0,0,5],parentObjectUniqueId=self.robotId, parentLinkIndex=self.robot.base_index)
 
    def reset_obj(self):
        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"29-0/q.npy"))
        self.data_gripper = np.load(os.path.join(self.robot_recordings_dir,"29-0/gripper.npy"))#np.load (self.env_root + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "29-1/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "29-1/gripper.npy"))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[0],glist[0])
        for i in range(0,5,1):
            glist[i] = max(220,glist[i])
            qlist[i][6] = 1.5
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

      
        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()#np.array(self.obj_position)
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[0] -= 0.01
        cur_pos[1] += 0.2
        cur_pos[2] += 0.05
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=130)
 
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]



    def get_success (self,seg=None):
        cameraEyePosition = np.array([0.4,0.0,2.0])#self.box_pos#viewMatrix[:3,3] 
        cameraUpVector = np.array([1,0,0])#viewMatrix[:3,1] * -1.0
        cameraTargetPosition = np.array([0.4,0.0,0.0])#viewMatrix[:3,3] - np.array()#+ viewMatrix[:3,2] * 0.001
        self._top_view_matrix = self.p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)

        img = self.p.getCameraImage (width=320,
                                         height=240,
                                         viewMatrix=self._top_view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

        im = img[2]
        hsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,(25,200,72),(86,255,255))
        green = np.zeros_like(im,np.uint8)
        dist = np.sum(mask)
        print(dist)
        if 0:
          plt.figure(0)
          plt.imshow(img[2])
          plt.figure(1)
          plt.imshow(green)
          plt.figure(2)
          plt.imshow(hsv)
          plt.figure(3)
          plt.imshow(mask)
          plt.show()

        gripper_pos = self.robot.getEndEffectorPos()
        if dist > (40855) and gripper_pos[2] < 0.45:#rot_deg < -40 and rotnorm/np.pi * 180 > 80:
          return True
        else:
          return False
