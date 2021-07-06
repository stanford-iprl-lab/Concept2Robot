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

from scipy.spatial.transform import Rotation as R

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine13(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine13,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    def init_obj(self):
        self.obj_file = os.path.join(self.urdf_dir, "objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.055, -0.065, 0.35]
        self.obj_scaling = 2.2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.4])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id


        self.box_file = os.path.join (self.urdf_dir, "openbox/openbox.urdf")
        self.box_position = [0.35, 0.03, -0.33]
        self.box_scaling = 0.000225
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
        gripper_v = 120

        for i in range(40,180,1):
            glist[i] = min(gripper_v,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        for _ in range(2):
          pos = self.robot.getEndEffectorPos()
          pos[2] += 0.05
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

        rotation_degree = np.random.uniform(-math.pi,math.pi)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)

        self.box_new_orientation = R.from_dcm(NewHTrans[:3,:3]).as_quat()

        self.p.resetBasePositionAndOrientation(self.box_id,self.box_new_position,self.box_new_orientation)


        ##### obj is reset
        cur_joint = self.robot.getJointValue()
        cur_pos = np.array(self.box_new_position)#self.robot.getendeffectorpos()
        cur_pos[:2]  = cur_pos[:2] * 0.5 + self.robot.getEndEffectorPos()[:2] * 0.5
        cur_orn = self.robot.getEndEffectorOrn()
        cur_pos[2] = self.robot.getEndEffectorPos()[2]
        print("before initialiaztion")
        for i in range(19):
          self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=gripper_v)

    #    input("raw")
        ####
    #    cur_joint = self.robot.getJointValue()
    #    cur_pos = np.array(self.box_new_position) + self.robot.getendeffectorpos()
    #    cur_orn = self.robot.getEndEffectorOrn()
    #    pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
    #    cur_pos[:2] += pos_diff
    #    cur_pos[2] = self.robot.getEndEffectorPos()[2] + np.random.uniform(-0.04,0.01)
    #    print("before initialiaztion")
    #    for i in range(19):
    #      self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=gripper_v)

#        cur_pos[2] += np.random.uniform(-0.05,0.05)

        for _ in range(20):
          self.p.stepSimulation()


    def init_grasp_(self):


        self.box_position[2] = -0.33
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(100,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[2] += 0.05
        orn = self.robot.getEndEffectorOrn()

        for i in range(109):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=100)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

        box_diff = np.random.uniform(-0.1,0.1,size=(2,))
        self.box_position[2] *= -1.0
        self.box_pos =  np.array([self.box_position[0]+box_diff[0],self.box_position[1]+box_diff[1],self.box_position[2]])
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_pos,self.box_orientation)

        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[0] =  self.box_position[0]
        cur_pos[1] = self.box_position[1]
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] += 0.02
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=120)


    def get_success(self,seg=None):
        box = self.p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        box_center = np.array(box_center)
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        obj_center = np.array(obj_center)
        dist = np.linalg.norm(box_center - obj_center)
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        box_obj_closet_info = self.p.getContactPoints(self.obj_id, self.box_id) 
        #print(len (left_closet_info),len (right_closet_info),obj[0][0], box[1][0])
        if len (left_closet_info) > 0 and len (right_closet_info) > 0:
          if dist < 0.05 or len(box_obj_closet_info) > 0:
            return True
          else:
            return False
        else:
          return False
