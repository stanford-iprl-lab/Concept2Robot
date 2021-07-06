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


class Engine172(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine172,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14
        self.p.setPhysicsEngineParameter(useSplitImpulse=True,splitImpulsePenetrationThreshold=0.01)
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

        self.load_model()

    def init_obj(self):
        self.obj_pos = [0.37,0.0,0.30]
        self.obj_x, self.obj_y, self.obj_z = self.obj_pos
        self.obj_ori = self.p.getQuaternionFromEuler([0.0,math.pi,-math.pi/2.0])
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/books/b1/b1.urdf"),self.obj_pos,self.obj_ori,useFixedBase=True)
        self.obj_joint_start = 2.5
        self.p.resetJointState(self.obj_id,0,self.obj_joint_start)
   
    def reset_obj(self):
        r = R.from_quat(self.obj_ori)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()
        HTrans[:3,3] = self.obj_pos

        rotation_degree = np.random.uniform(-0.3,0.3)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)

        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj1_pos = np.copy(self.obj_pos)
        self.obj1_pos[:2] = self.obj1_pos[:2] + pos_diff

        self.obj1_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj1_pos,self.obj1_ori)

        self.p.resetJointState(self.obj_id,0,self.obj_joint_start)
  
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
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.initial_pos = self.data_q[0]#(-1.3026999182595653, -1.210032113999055, 0.79519250956187, -2.118622450107143, 0.8971789146016195, 1.0616185345092588, -0.34515004476469724)
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

    def init_grasp(self):
        self.robot.gripperControl(0)
        self.robot.setJointValue(self.initial_pos,220)

        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], 0.0)

        for init_t in range(100):
            box = self.p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.15
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        self.null_q = self.robot.getJointValue()#self.initial_pos

        self.obj_x, self.obj_y, self.obj_z = self.obj_pos
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        pos = [self.obj_x-0.03+transl[0],self.obj_y-0.25+transl[1],self.obj_z+0.0]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,seg=None):
        jointInfo = self.p.getJointState(self.obj_id,0)
        if jointInfo[0] < 1.8:
          return True
        else:
          return False
