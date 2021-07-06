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

class Engine85(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine85,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    def init_obj(self):
        self.box_file = os.path.join (self.urdf_dir, "openbox2/openbox.urdf")
        print("self.box_file",self.box_file)
        self.box_position = [0.40, 0.02, 0.34]
        self.box_scaling = 0.00035
        self.box_orientation = self.p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.07, 0.35]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.4])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id
        self.pos = None

        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=0.01)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.9)

        
    def reset_obj(self):
       self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)



    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.reset_obj()
        self.box_position[2] = -.34
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        gripper_v = 130

        for i in range(40, 180, 1):
            glist[i] = min(gripper_v, glist[i])
            self.robot.jointPositionControl(qlist[i], gripper=glist[i])

        for _ in range(1):
            pos = self.robot.getEndEffectorPos()
            pos[2] += 0.05
            orn = self.robot.getEndEffectorOrn()
            for i in range(30):
                self.robot.positionControl(pos, orn, null_pose=self.null_q, gripperPos=gripper_v)

        pos = self.robot.getEndEffectorPos()
        pos[1] += 0.05
        pos[0] += 0.1
        pos[2] += 0.0
        orn = self.robot.getEndEffectorOrn()
        #self.robot.moveTo(startPos=self.robot.getEndEffectorPos(),goalPos=pos,orn=orn, jointValues=self.null_q, gripperV=gripper_v)
        for i in range(10):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=140)
        time.sleep(3)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.box_position[2] *= -1.0
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)
        self.pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]

 
    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj_pos[1] < self.pos[1] - 0.05:
          return True
        else:
          return False

