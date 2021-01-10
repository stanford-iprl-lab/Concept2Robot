#!/usr/bin/env python3

"""
action 101: push sth with sth

TODO : recover these functions: get_reward
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


class Engine101(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine101,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 30.0)
        self.p.setGravity(0,0,-9.81)

    def reset_table(self):
        self.p.changeVisualShape(self.table_id,-1,textureUniqueId=self.table_textid,specularColor=[0,0,0])

    def init_obj(self):
        self.obj_position = [0.42, -0.11, 0.333]
        self.obj_scaling = 1.5
        self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b3/b3.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        texture_path = os.path.join(self.resources_dir,'textures/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = self.p.loadTexture(texture_file)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])

        self.obj2_position = [0.42, 0.09, 0.333]
        self.obj2_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.obj2_scaling = 1.5
        self.obj2_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b3/b3.urdf"),basePosition=self.obj2_position,baseOrientation=self.obj2_orientation,globalScaling=self.obj_scaling)

        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[0.1,.3,1.,1])

    def reset_obj(self):
        obj_x = 0.42
        obj_y = -0.11
        transl = np.random.uniform(-0.1,0.1,size=(2,)) 
        self.obj_position = np.array([obj_x+0.0, obj_y+0.0, 0.333])
 
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        self.p.resetBasePositionAndOrientation(self.obj2_id,self.obj2_position,self.obj2_orientation)
        
        obj_friction_ceof = 0.3
        self.p.changeDynamics(self.obj_id, -1, mass=1.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=100.0)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=100.0)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=10.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=10.0, contactDamping=0.9)

        obj2_friction_ceof = 0.4
        self.p.changeDynamics(self.obj2_id, -1, mass=1.9)
        self.p.changeDynamics(self.obj2_id, -1, lateralFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, rollingFriction=100.0)
        self.p.changeDynamics(self.obj2_id, -1, spinningFriction=100.0)
        self.p.changeDynamics(self.obj2_id, -1, linearDamping=20.0)
        self.p.changeDynamics(self.obj2_id, -1, angularDamping=10.0)
        self.p.changeDynamics(self.obj2_id, -1, contactStiffness=10.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=10000.0, contactDamping=0.9)

        for i in range(4):
          self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
          self.p.resetBasePositionAndOrientation(self.obj2_id,self.obj2_position,self.obj2_orientation)
 
    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"101-9/q.npy"))
        self.data_gripper = np.load (os.path.join(self.robot_recordings_dir,"101-9/gripper.npy"))
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])


    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "101-9/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "101-9/gripper.npy"))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[1],glist[1])
        for i in range(1,30,1):
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        
        self.pos = self.p.getBasePositionAndOrientation(self.obj2_id)[0]

    def taskColliDet(self):
        colli = False
        for y in [0,1,2,3,4,5,6]:
          c = self.p.getContactPoints(bodyA=self.obj_id,bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
          c2 = self.p.getContactPoints(bodyA=self.obj2_id,bodyB=self.robotId,linkIndexB=y)
          if len(c) > 0:
            colli = True
            return True
        return False

    def get_success(self,seg=None):
        pos = np.array(self.p.getBasePositionAndOrientation(self.obj2_id)[0])
        dist = np.linalg.norm(pos-self.pos)
        contact_info = self.p.getContactPoints (self.robotId, self.obj_id)
        if len(contact_info) > 0:
          self.contact = True
        if dist > 0.03 and self.contact and dist < 0.1:
          return True
        else:
          return False

