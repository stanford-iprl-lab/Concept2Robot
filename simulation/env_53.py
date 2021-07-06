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

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


class Engine53(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine53,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
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
        self.objId_list = []
        print("initiializaing !")
        radius = 0.025
        self.obj_position_list = []
        self.obj_scaling = 1.0
        self.obj_color_list = []
        self.obj_color_list.append([0., 1., 0.,1.0])
        self.obj_color_list.append([0.949, 0.878, 0.0392, 1.0])
        self.obj_color_list.append([0.12156, 0.3804, 0.6745, 1.0])
        self.obj_color_list.append([0.9254901, 0.243137, 0.086274509, 1.0])
        for i in range(4):
           self.obj_position_list.append([0.4,-0.05,0.32+radius+i*2*radius])
           objId = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/cubes/c1/cube_small.urdf"),self.obj_position_list[i])
           self.objId_list.append(objId)
           self.p.changeVisualShape (objId, -1, rgbaColor=self.obj_color_list[i])
 
    def reset_obj(self):
        print("resetting")
        obj_x = 0.4
        obj_y = -0.02
        transl = np.random.uniform(-0.1,0.1,size=(2,)) 
        self.obj_x = obj_x + transl[0]
        self.obj_y = obj_y + transl[1]  
        for i in range(4):
          self.obj_position_list[i][0] = self.obj_x
          self.obj_position_list[i][1] = self.obj_y
          self.p.resetBasePositionAndOrientation(self.objId_list[i],self.obj_position_list[i],self.p.getQuaternionFromEuler([0,0,0]))
        for i in range(50):
          self.p.stepSimulation()

    def reset_obj_2(self):
        print("resetting")
        for i in range(len(self.objId_list)):
          self.p.resetBasePositionAndOrientation(self.objId_list[i],self.obj_position_list[i],self.p.getQuaternionFromEuler([0,0,0]))
        for i in range(50):
          self.p.stepSimulation()

    def init_grasp(self):
        self.robot.gripperControl(255)    
  
        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
        self.null_q = self.robot.getJointValue()

        self.obj_x, self.obj_y, self.obj_z = self.obj_position_list[2]
        pos = [self.obj_x-0.01,self.obj_y-0.2,self.obj_z+0.0]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=255)
       
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.reset_obj_2()
        self.contact = False
        self.init_obj_pos = self.p.getBasePositionAndOrientation(self.objId_list[3])[0]


    def taskColliDet(self):
        colli = False
        for y in [0,1,2,3,4,5,6]:
         for i in range(len(self.objId_list)):
          c = self.p.getContactPoints(bodyA=self.objId_list[i],bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
        return False

    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.objId_list[3])[0]
        contact_info = self.p.getContactPoints (self.table_id, self.objId_list[3])
        if len(contact_info) > 0:
          return True
        else:
          return False   
