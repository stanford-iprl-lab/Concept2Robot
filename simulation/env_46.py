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

from scipy.spatial.transform import Rotation as R

import sys
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

class Engine46(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine46,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14
        self.p.setPhysicsEngineParameter(useSplitImpulse=True,splitImpulsePenetrationThreshold=0.01)
        self.load_model()

        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

        self.count = 0
        self.fw1 = open("p1.txt","w")
        self.fw2 = open("p2.txt","w")


    def init_obj(self):
        self.obj_id = self.p.loadURDF(fileName=os.path.join(self.urdf_dir,"obj_libs/drawers/d4/d4.urdf"),useFixedBase=True)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        self.p.changeVisualShape (self.obj_id, 0, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 1, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 2, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 3, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 4, rgbaColor=[0,0,1,1.0])
 
        self.p.resetJointState(self.obj_id,0,0.05) 
  
        numJoint = self.p.getNumJoints(self.obj_id) 
        for jointIndex in range(numJoint):
            jointInfo = self.p.getJointInfo(self.obj_id,jointIndex)
            print(jointInfo)
       

    def reset_obj(self):
        self.obj_x = 0.38
        self.obj_y = 0.05
        self.obj_z = 0.35

        self.obj1_ori =  self.p.getQuaternionFromEuler ([math.pi/2.0,0 ,-math.pi/2.0 + 0.1])

        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_pos_new = np.array([self.obj_x+transl[0],self.obj_y+transl[1],self.obj_z])

        r = R.from_quat(self.obj1_ori)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()

        rotation_degree = np.random.uniform(-0.5,0.5)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)
        self.obj1_ori_new = R.from_dcm(NewHTrans[:3,:3]).as_quat()

        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_pos_new,self.obj1_ori_new)
        init_d = np.random.uniform(0,0.04)
        self.p.resetJointState(self.obj_id,0,init_d)
        
        obj_friction_ceof = 0.3
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=0.3)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=100.0)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=100.0)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=10000.0, contactDamping=1)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=10000.0, contactDamping=0.01)


    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-1/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.initial_pos = (-1.3026999182595653, -1.210032113999055, 0.79519250956187, -2.118622450107143, 0.8971789146016195, 1.0616185345092588, -0.34515004476469724)
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

    def init_grasp(self):
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

        self.null_q = self.initial_pos
 
        obj_x, obj_y, obj_z = self.obj_pos_new
        pos = [obj_x+0.03,obj_y+0.3,obj_z+0.3]
        orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)
        pos = [obj_x+0.0,obj_y+0.0,obj_z+0.3]
        orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        pos = [obj_x-0.05,obj_y+0.0,obj_z+0.22]
        orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)
 
        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,suc=None):
        jointInfo = self.p.getJointState(self.obj_id,0)
        if jointInfo[0] > 0.1:
          return True
        else:
          return False

