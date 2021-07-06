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
sys.path.append('./Eval')
sys.path.append('./')

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine8(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine8,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    
    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir,"objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.07, 0.35]
        self.obj_scaling = 2.2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.4])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        self.box_file = os.path.join (self.urdf_dir, "openbox5/openbox.urdf")
        self.box_position = [0.35,0.1,-0.34]#[0.39, 0.00, -0.34]
        self.box_scaling = 1.0#0.8#0.00035
        self.box_orientation = self.p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[38/255.,0.,128/255.0,1])
        self.p.changeDynamics(self.obj_id,-1,mass=2.0)

        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=0.01)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.9)

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id, self.obj_position, self.obj_orientation)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        gripper_v = 130

        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

        for i in range(40,180,1):
            glist[i] = min(gripper_v,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        for _ in range(1):
          pos = self.robot.getEndEffectorPos()
          pos[2] += 0.15
          orn = self.robot.getEndEffectorOrn()
          for i in range(30):
            self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gripper_v)

        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]


        ######## obj is lifted

        transl = np.random.uniform(-0.05,0.05,size=(2,))
        self.box_new_position = np.array(self.box_position)
        self.box_new_position[2] *= -1.0
        self.box_new_position[:2] += transl

        r = R.from_quat(self.box_orientation)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()

        rotation_degree = 0.0#np.random.uniform(-math.pi,math.pi)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)

        self.box_new_orientation = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        #self.box_new_orientation = self.box_orientation
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_new_position,self.box_new_orientation)


        ##### obj is reset
        cur_joint = self.robot.getJointValue()
        cur_pos = np.array(self.box_new_position)#self.robot.getendeffectorpos()
        cur_orn = self.robot.getEndEffectorOrn()
        box_AABB = self.p.getAABB(self.box_id)
        cur_pos[0] = box_AABB[0][0] * 0.5 + box_AABB[1][0] * 0.5 - 0.05
        cur_pos[1] = box_AABB[1][1] * 0.5 + box_AABB[0][1] * 0.5 - 0.15
        cur_pos[2] = self.robot.getEndEffectorPos()[2]
        print("before initialiaztion")
        for i in range(4):
          self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=gripper_v)

        ####
        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.15,0.15,size=(2,))
        #cur_pos[:2] += pos_diff
        cur_pos[0] += -0.1
        cur_pos[1] += -0.1
        cur_pos[2] += np.random.uniform(-0.1,0.1)
        #cur_pos[2] += self.robot.getEndEffectorPos()[2] + np.random.uniform(-0.05,0.05)
        print("before initialiaztion")
        for i in range(4):
          self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=gripper_v)


    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]

        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info)==0 and len (right_closet_info)==0 and obj[0][0] > box[1][0]:
          return True
        else:
          return False
