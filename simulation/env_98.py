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


class Engine98(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine98,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
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
        self.obj_position = [0.42, -0.08, 0.43]
        self.obj_scaling = 1.0
        self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.resources_dir,"urdf/obj_libs/bottles/b3/b3.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.,0.2,0.8,1])

        self.box_file = os.path.join (self.resources_dir, "urdf/obj_libs/cubes/c4/c4.urdf")
        self.box_position = [0.44, -0.05, 0.35]
        self.box_scaling = 1.
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.box_id, -1, rgbaColor=[1.,0.0,0.0,1])


    def reset_obj(self):
        box_x = 0.44
        box_y = -0.05
        transl = np.random.uniform(-0.05,0.05,size=(2,))
        self.box_position = [box_x + transl[0],box_y + transl[1], 0.33]
        self.p.resetBasePositionAndOrientation (self.box_id, self.box_position, self.box_orientation)

        obj_x = self.box_position[0]
        obj_y = self.box_position[1]
        transl_obj = np.random.uniform(-0.05,0.05,size=(2,))
        self.obj_position = [obj_x + transl_obj[0],obj_y + transl_obj[1], 0.43]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)

        #self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        obj_friction_ceof = 1000.0
        self.p.changeDynamics(self.obj_id, -1, mass=1.0)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=100.0)
        #self.p.changeDynamics(self.obj_id, -1, contactStiffness=1000.0, contactDamping=0.9)

        #table_friction_ceof = 1.0
        #self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        #self.p.changeDynamics(self.table_id, -1, contactStiffness=1000.0, contactDamping=10.0)

        for i in range(5):
          self.p.stepSimulation()

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
        pos[0] += -.1
        pos[2] += .2
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=0)

        cur_joint = self.robot.getJointValue()
        cur_pos = np.array(self.obj_position)
        cur_pos[0] -= 0.1
        cur_pos[1] -= 0.18#self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=0)
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

        for _ in range(30):
          self.p.stepSimulation()
        self.pos = self.p.getBasePositionAndOrientation(self.obj_id)[0][2]
        self.contact = False

    def get_success(self,seg=None):
        pos = self.p.getBasePositionAndOrientation(self.obj_id)[0][2]
        closet_info = self.p.getContactPoints (self.robotId, self.obj_id)
        if len(closet_info) > 0:
          self.contact = True
        if np.abs(pos-self.pos) < 0.04 and self.dmp.timestep >= self.dmp.timesteps and self.contact:
          return True
        else:
          return False

