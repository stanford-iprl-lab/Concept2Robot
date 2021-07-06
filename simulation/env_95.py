#!/usr/bin/env python3
"""
    action101: put sth with sth
"""
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
from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine95(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine95,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self.wid = worker_id
        self.p = p_id
        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        #self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

        self.pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        self.orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        self.box_file = os.path.join (self.urdf_dir, "obj_libs/cubes/c3/c3.urdf")
        self.box_position = [0.48, -0.05, 0.27]
        self.box_scaling = 1.5
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

    def init_obj(self):
        self.obj_file = os.path.join(self.resources_dir, "urdf/objmodels/nut.urdf")
        self.obj_position = [0.4, -0.15, 0.34]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi / 2, -math.pi / 2, 0])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,
                                      baseOrientation=self.obj_orientation,
                                      globalScaling=self.obj_scaling)  # ,physicsClientId=self.physical_id)
        self.p.changeVisualShape(self.obj_id, -1, rgbaColor=[0.3, 0.3, 0.9, 1])

    def reset_obj(self):
        box_x = 0.48
        box_y = -0.05
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.box_position = [box_x + transl[0],box_y + transl[1], 0.27]
        self.p.resetBasePositionAndOrientation (self.box_id, self.box_position, self.box_orientation)
        self.p.changeVisualShape (self.box_id, -1, rgbaColor=[1.0, 0.0, 0.0, 1])
 
        obj_x = self.box_position[0]
        obj_y = self.box_position[1]
        transl_obj = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_position = [obj_x + transl_obj[0],obj_y + transl_obj[1], 0.33]
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)



    def init_grasp(self):
        self.robot.gripperControl(0)
        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[2] += 0.1
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=150)

        cur_joint = self.robot.getJointValue()
        cur_pos = np.array(self.obj_position)#self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        cur_pos[1] += -0.2
        cur_pos[0] += -0.08
        cur_pos[2] += 0.1
        for i in range(109):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=0)

        pos = self.robot.getEndEffectorPos()
        pos[2] -= 0.02
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=0)
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)

        for _ in range(100):
          self.p.stepSimulation()

    def get_success(self,seg=None):
        closet_info = self.p.getContactPoints (self.box_id, self.obj_id)
        box_AABB = self.p.getAABB(self.box_id)
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        if len(closet_info) == 0 and self.dmp.timestep >= self.dmp.timesteps:
          return True
        else:
          return False

