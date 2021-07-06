#!/usr/bin/env python3

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

from robot_peg import Robot
from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine49(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine49,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=Robot)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 500.0
        self.robot.armMaxForce = 500.0


    def init_obj(self):
        self.start_position = [0.25, 0, 0.48]

        self.box_w = 0.30
        self.box_h = 0.0
        self.box_d = 0.30

        self.hole_half_r = 0.015
        mass = 0

        self.box_center = [self.box_w, self.box_h, self.box_d]
        obj1_w = 0.05
        obj1_h = 0.02
        obj1_d = 0.025
        self.obj1_position = [self.box_w, self.box_h - self.hole_half_r - obj1_h, self.box_d]
        self.obj1_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj1_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj1_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj1_id = self.p.createMultiBody(mass, self.obj1_c, self.obj1_v, self.obj1_position)
        self.p.changeVisualShape (self.obj1_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])

        self.obj2_position = [self.box_w, self.box_h + self.hole_half_r + obj1_h, self.box_d]
        self.obj2_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj2_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj2_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj2_id = self.p.createMultiBody(mass, self.obj2_c, self.obj2_v, self.obj2_position)
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])

        obj3_w = (obj1_w - self.hole_half_r) / 2.0
        obj3_h = self.hole_half_r
        obj3_d = obj1_d
        self.obj3_position = [self.box_w - self.hole_half_r - obj3_w, self.box_h, self.box_d]
        self.obj3_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj3_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj3_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj3_id = self.p.createMultiBody(mass, self.obj3_c, self.obj3_v, self.obj3_position)
        self.p.changeVisualShape (self.obj3_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])

        self.obj4_position = [self.box_w + self.hole_half_r + obj3_w, self.box_h, self.box_d]
        self.obj4_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj4_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj4_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj4_id = self.p.createMultiBody(mass, self.obj4_c, self.obj4_v, self.obj4_position)
        self.p.changeVisualShape (self.obj4_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])

        self.box_AABB = [self.box_w - self.hole_half_r - 2 * obj3_w, self.box_h - self.hole_half_r - 2 * obj3_h, self.box_d, self.box_w + self.hole_half_r + 2 * obj3_w, self.box_h + self.hole_half_r + 2 * obj3_h, obj1_d + self.box_d]


    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj2_id,self.obj2_position,self.obj2_orientation)
        
        obj2_friction_ceof = 2000.0
#        self.p.changeDynamics(self.obj2_id, -1, mass=0.05)
        self.p.changeDynamics(self.obj2_id, -1, lateralFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, rollingFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, spinningFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, linearDamping=20.0)
        self.p.changeDynamics(self.obj2_id, -1, angularDamping=1.0)

        self.p.changeDynamics(self.obj1_id, -1, contactStiffness=10.0, contactDamping=0.1)
        self.p.changeDynamics(self.obj2_id, -1, contactStiffness=10.0, contactDamping=0.1)
        self.p.changeDynamics(self.obj3_id, -1, contactStiffness=10.0, contactDamping=0.1)
        self.p.changeDynamics(self.obj4_id, -1, contactStiffness=10.0, contactDamping=0.1)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1000.0, contactDamping=0.1)


    def init_motion(self):
        self.initial_pos = (-1.2753665984589995, -1.229056745895614, 0.6844703519077765, -2.3988362862274095, 0.6620342319319344, 1.3214016490535152, -0.8183599835108928)
        self.null_q = self.initial_pos
        self.robot.setJointValue(self.initial_pos,220)
        self.data_q = []
        self.data_q.append(self.null_q)

    def init_grasp(self):
        initial_pos = (-1.2753665984589995, -1.229056745895614, 0.6844703519077765, -2.3988362862274095, 0.6620342319319344, 1.3214016490535152, -0.8183599835108928)
        self.null_q = initial_pos

        pos = [0.28,0.03,0.538]
        orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        self.robot.setJointValue(initial_pos,220)

        for i in range(19):
          self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        self.p.setTimeStep(1 / 500.0)
        #pos = [0.30,0.01,0.533]
        #orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        #for i in range(19):
        #    self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

    def get_success(self,seg=None):
        success_flag = False
        current_pos = self.robot.getPegPos()
        dist = np.linalg.norm(current_pos[0] - self.box_center[0]) + np.linalg.norm(current_pos[1]-self.box_center[1]) #+ np.linalg.norm(current_pos[2]-0.517)
        if current_pos[2] < self.box_AABB[2] + 0.01 and current_pos[0] > self.box_AABB[0] and current_pos[0] < self.box_AABB[3] and current_pos[1] > self.box_AABB[1] and current_pos[1] < self.box_AABB[4]:
          success_flag = True
        return success_flag
 
