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

from scipy.spatial.transform import Rotation as R

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

class Engine5(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine5,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14
        self.p.setPhysicsEngineParameter(useSplitImpulse=True, splitImpulsePenetrationThreshold=0.01)
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG, globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)


    def init_obj(self):
        self.obj_scaling = 0.8
        self.obj_id = self.p.loadURDF(fileName=os.path.join(self.urdf_dir,"obj_libs/drawers/d4/d4.urdf"),useFixedBase=True,globalScaling=self.obj_scaling)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        self.p.changeVisualShape (self.obj_id, 0, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 1, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 2, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 3, rgbaColor=[0,0,1,1.0])
        self.p.changeVisualShape (self.obj_id, 4, rgbaColor=[0,0,1,1.0])

        open_deg = np.random.uniform(0.05,0.12,size=(1,))
        self.p.resetJointState(self.obj_id,0,open_deg)
   
    def reset_obj(self):
        self.obj_x = 0.43
        self.obj_y = 0.0#-0.07
        self.obj_z = 0.34
        self.obj1_pos = np.array([self.obj_x,self.obj_y,self.obj_z])
        self.obj1_ori =  self.p.getQuaternionFromEuler ([math.pi/2.0,0 ,-math.pi/2.0 + 0.1])

        r = R.from_quat(self.obj1_ori)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm() 
        HTrans[:3,3] = self.obj1_pos
       
        rotation_degree = np.random.uniform(-0.5,0.5)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)
 
        self.obj1_pos = NewHTrans[:3,3]
        transl = np.random.uniform(-0.05,0.05,size=(2,))
        self.obj1_pos[0] += transl[0]
        self.obj1_pos[1] += transl[1]
 
        self.obj1_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj1_pos,self.obj1_ori)

        open_deg = np.random.uniform(0.05,0.1,size=(1,))
        self.p.resetJointState(self.obj_id,0,open_deg)
  
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
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-0/q.npy"))
        self.data_gripper = np.load ( os.path.join(self.configs_dir,'init/gripper.npy'))
        self.initial_pos = self.data_q[0]#(-1.3026999182595653, -1.210032113999055, 0.79519250956187, -2.118622450107143, 0.8971789146016195, 1.0616185345092588, -0.34515004476469724)
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

    def init_grasp(self):
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

        self.null_q = self.initial_pos

        pos = [0.3,0.0,0.51]
        orn = self.p.getQuaternionFromEuler([math.pi+0.5,0,0.])
        for i in range(30):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)


        pos = [0.17,0.0,0.35]
        orn = self.p.getQuaternionFromEuler([math.pi+0.5,0,0.])
        for i in range(30):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        transl = np.random.uniform(-0.05,0.05,size=(2,))
        pos = self.robot.getEndEffectorPos()
        pos[:2] += transl
        for i in range(30):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        self.reset_obj()

        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def taskColliDet(self):
        colli = False
        for y in [0, 1, 2, 3, 4]:
            c = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.robotId, linkIndexB=y)
            # cl = self.p.getClosestPoints(bodyA=self.robotId,bodyB=self.robotId,distance=100,linkIndexA=x,linkIndexB=y)
            if len(c) > 0:
                colli = True
                print("colli",colli, y)
                return True
        return False

    def get_success(self,suc=None):
        jointInfo = self.p.getJointState(self.obj_id,0)
        if self.taskColliDet():
            print("collision detected!")
            #time.sleep(20)
            return False

        if jointInfo[0] < 0.01:
          p1_ori = self.p.getLinkState (self.obj_id, 0)[1]
          rr = R.from_quat(p1_ori)
          HTrans = np.zeros((4,4))
          HTrans[:3,:3] = rr.as_dcm()
          p1_pos = self.p.getLinkState (self.obj_id, 0)[0]
          p2_pos = p1_pos - HTrans[:3,0] * jointInfo[0]
          return True
        else:
          return False
