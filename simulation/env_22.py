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


class Engine22(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine22,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)


    def init_obj(self):
        self.obj_position = np.array([0.35, -0.02, 0.36])
        self.obj_scaling = 1.0
        self.obj_orientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/cylinders/c1/c1.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        obj_friction_ceof = 1000.0
        self.p.changeDynamics(self.table_id, -1, mass=0.0)
        self.p.changeDynamics(self.table_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=obj_friction_ceof)


        obj_friction_ceof = 1.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=20.0)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=1.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.9)

 
    def reset_obj(self):
        r = R.from_quat(self.obj_orientation)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()
        HTrans[:3,3] = self.obj_position
        print(HTrans)

        rotation_degree = np.random.uniform(-0.3,0.3)
        print("rotation_degree",rotation_degree)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)
        print("NewHTrans",NewHTrans)

        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj1_pos = np.copy(self.obj_position)
        self.obj1_pos[0] += transl[0]
        self.obj1_pos[1] += transl[1]
        self.obj1_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj1_pos,self.obj1_ori)
        
        obj_friction_ceof = 1000.0
        self.p.changeDynamics(self.obj_id, -1, mass=1.0)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        for i in range(5):
          self.p.stepSimulation()

    def init_grasp(self):
        self.robot.gripperControl(255)    
  
        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = self.p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        self.null_q = self.robot.getJointValue()#self.initial_pos
    
        self.obj_x, self.obj_y, self.obj_z = self.obj_position
        pos = [self.obj_x-0.03,self.obj_y-0.25,self.obj_z+0.0]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=220)

        start_id = 0
        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        print("before initialiaztion")
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=135)

        for i in range(100):
          self.p.stepSimulation()

        self.init_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]

        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success (self,seg=None):
        pos1 = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        pos1 = np.array(pos1)
        dist = np.linalg.norm(pos1-self.init_pos)
        #print("vec",vec,"angle",np.arccos(np.abs(vec[2]))/np.pi * 180.0)
        if dist > 0.1:
          return True
        else:
          return False


