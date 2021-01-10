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
    from env_8 import Engine8
    from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine12(Engine8):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine12,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.07, 0.35]
        self.obj_scaling = 2.2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.4])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        self.box_file = os.path.join (self.resources_dir, "urdf/obj_libs/cubes/c3/c3.urdf")
        self.box_position = [0.42, -0.02, -0.27]
        self.box_scaling = 1.5
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
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

    def get_success(self,seg):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])][:2]
        box_center = np.array(box_center)
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])][:2]
        obj_center = np.array(obj_center)
        dist = np.linalg.norm(box_center - obj_center)
        box_corner = np.array([box[0][0], box[0][1]])
        box_radius = np.linalg.norm(box_corner - box_center)
        closet_info = self.p.getContactPoints (self.box_id, self.obj_id)
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info)==0 and len (right_closet_info)==0 and len(closet_info) > 0:
          return True
        else:
          return False

