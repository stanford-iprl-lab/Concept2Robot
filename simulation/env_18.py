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

from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('./')

from env_16 import Engine16
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


class Engine18(Engine16):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine18,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti
   

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.07, 0.35]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.4])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        self.box_file = os.path.join (self.urdf_dir, "openbox5/openbox.urdf")
        self.box_position = [0.43, 0.00, -0.34]
        self.box_scaling = 0.7#0.00035
        self.box_orientation = self.p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[38/255.,0.,128/255.0,1])
        self.p.changeDynamics(self.obj_id,-1,mass=2.0)

        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=0.01)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.9)


  
    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]

        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        #print(len (left_closet_info),len (right_closet_info),obj[0][0], box[1][0])
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj[1][0] < box[0][0]:
          return True
        else:
          return False
