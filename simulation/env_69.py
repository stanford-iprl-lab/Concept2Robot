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

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


class Engine69(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine69,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        self.p = p_id
#        self.p.setPhysicsEngineParameter(enableConeFriction=1)
#        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
#        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

#        self.p.setPhysicsEngineParameter(numSolverIterations=20)
#        self.p.setPhysicsEngineParameter(numSubSteps=10)

#        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
#        self.p.setPhysicsEngineParameter(enableFileCaching=0)
        #self.p.setTimeStep(1 / 30.0)
#        self.p.setGravity(0,0,-9.81)
        self.pos = None

    def init_obj(self):
        self.obj_position = [0.42, 0.04, 0.34]
        self.obj_scaling = 1.0
        self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b3/b3.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])

    def reset_obj(self):
        obj_x = 0.42
        obj_y = -0.01
        transl = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj_x = obj_x + transl[0]
        self.obj_y = obj_y + transl[1]
        self.obj_position = np.array([self.obj_x,self.obj_y,0.33])
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

        obj_friction_ceof = 1000.0
        self.p.changeDynamics(self.obj_id, -1, mass=1.0)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=100.0)

        table_friction_ceof = 1.0
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=1.0)

        for i in range(5):
          self.p.stepSimulation()

    def init_grasp(self):
        self.robot.gripperControl(255)    

        pos_traj = np.load (os.path.join (self.configs_dir, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        self.null_q = self.robot.getJointValue()
        self.obj_x, self.obj_y, self.obj_z = self.obj_position
        pos = [self.obj_x-0.1,self.obj_y-0.25,self.obj_z+0.1]
        orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=255)

        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        self.contact = False

    def taskColliDet(self):
        colli = False
        for y in [0,1,2,3,4,5,6]:
          c = self.p.getContactPoints(bodyA=self.obj_id,bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
        return False

    def get_success(self,seg=None):
        pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        gripper_pos = self.robot.getGripperTipPos()
        dist = np.linalg.norm(pos-gripper_pos)
        contact_info = self.p.getContactPoints (self.robotId, self.obj_id)
        print("dist",dist)
        if len(contact_info) > 0:
          self.contact = True
        if dist < 0.13 and not self.contact:
          return True
        else:
          return False
