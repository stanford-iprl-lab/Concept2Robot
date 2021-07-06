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
from robot_scoop import Robot
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

#################################
import signal
import importlib
import torch
import torch.nn as nn

import sh
import re
import torch.nn.functional as F
np.set_printoptions(precision=4,suppress=True,linewidth=300)


class Engine123(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine123,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=Robot)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 10000.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14
    
    def reset_new(self):
        self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
        return self.get_observation()

    def init_obj(self):
        self.mlist = []
        self.mori = []
        self.mpos = []
        self.obj_scaling = 1.8
        self.obj_pos = [0.3437 , -0.04, 0.30]
        self.obj_ori = [0, 0, -0.1494381, 0.9887711]
        self.obj_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/boxes/b2/b2.urdf"),basePosition=self.obj_pos,baseOrientation=self.obj_ori,globalScaling=self.obj_scaling,useFixedBase=True)
        for i in range(-1,4):
          self.p.changeVisualShape (self.obj_id, i, rgbaColor=[1.,0.,0.,1])
      
        print("init_obj is calling")
        self.num_boxes = 9
        for i in range(self.num_boxes):
          if i < int(round(self.num_boxes / 3.)):
            obj2_pos = [0.3137, 0.06, 0.35 + i * 0.08]
          elif i < int(round(self.num_boxes * 2./3.)):
            # obj2_pos = [0.3437, 0.05, 0.35 + i * 0.03]
            obj2_pos = [0.3437, 0.0, 0.35 + (i - round(self.num_boxes/3.)) * 0.08]
          else:
            # obj2_pos = [0.3437, 0.05, 0.35 + i * 0.03]
            obj2_pos = [0.3837, 0.04, 0.35 + (i - round(self.num_boxes*2./3.)) * 0.08]
          obj2_ori = self.obj_ori
          obj2_ori = np.random.rand(4)
          obj2_ori /= np.linalg.norm(obj2_ori)
          obj2_ori = obj2_ori.tolist()
          # obj2_pos = [0.3437,0.04,0.35]
          # obj2_ori = [0.,0.,0.,1.]
          self.mpos.append(obj2_pos) 
          self.mori.append(obj2_ori)
          obj2_id = self.p.loadURDF(os.path.join(self.resources_dir, "urdf/obj_libs/cubes/c2/c2.urdf"),
                                    basePosition=obj2_pos,baseOrientation=obj2_ori,
                                    globalScaling=0.75)
          friction_ceof = 2000.0
          # self.p.changeDynamics(obj2_id, -1, lateralFriction=friction_ceof)
          # self.p.changeDynamics(obj2_id, -1, rollingFriction=friction_ceof)
          # self.p.changeDynamics(obj2_id, -1, spinningFriction=friction_ceof)
          # self.p.changeDynamics(obj2_id, i, linearDamping=1000.0)
          # self.p.changeDynamics(obj2_id, i, angularDamping=1000.0)
          # self.p.changeDynamics(obj2_id, i, contactStiffness=0.1, contactDamping=0.1)
          self.p.changeVisualShape (obj2_id, -1, rgbaColor=[0.,0.,1.,1])
          self.mlist.append(obj2_id)
          # self.p.resetBasePositionAndOrientation(self.mlist[i],self.mpos[i],self.mori[i])

    def reset_obj(self):
        if len(self.mlist) > 0:
            for i in range(self.num_boxes):
                self.p.resetBasePositionAndOrientation(self.mlist[i],self.mpos[i],self.mori[i])

        # for i in range(5):
        # for i in range(300):
        #     print(i)
        #     self.p.stepSimulation()

        # positions = []
        # orientations = []
        # vs = []
        # ws = []
        # for obj_id in self.mlist:
        #     pos, ori = self.p.getBasePositionAndOrientation(obj_id)
        #     v, w = self.p.getBaseVelocity(obj_id)
        #     positions.append(pos)
        #     orientations.append(ori)
        #     vs.append(v)
        #     ws.append(w)
        # print(positions)
        # print(orientations)
        # print(vs)
        # print(ws)
        # input()

#        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_pos,self.obj_ori)
  
        obj_friction_ceof = 20000.0
 #       for i in range(-1,5):
 #         self.p.changeDynamics(self.obj_id, i, mass=0.09)
        table_friction_ceof = 0.4
        # self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        # self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        # self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        # self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        print("initializing")
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])

        gvalue = 240
        self.robot.setJointValue(qlist[0],240)#glist[0])
        # self.robot.setJointValue(qlist[140],glist[140])
        # print(len(qlist))
        for i in range(0,42,1):
            print("q: {}".format(qlist[i]))
            glist[i] = min(gvalue,glist[i])
            qlist[i][6] += 0.1
            self.robot.jointPositionControl(qlist[i],gripper=240)#glist[i])
            self.p.stepSimulation()
            print(i)
        #    input("step")

        self.null_q = self.robot.getJointValue()
        pos = self.robot.getEndEffectorPos()#[self.obj_x-0.2,self.obj_y-0.3,self.obj_z+0.3]
        orn = self.robot.getEndEffectorOrn()#$self.p.getQuaternionFromEuler([math.pi,0,0])
        pos[0] -= 0.1
        for i in range(19):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gvalue)

        #print(self.p.getEulerFromQuaternion(orn))
        #pos = self.obj_pos#self.robot.getEndEffectorPos()#[self.obj_x-0.2,self.obj_y-0.3,self.obj_z+0.3]
        #orn = self.p.getQuaternionFromEuler([3.0111602527051042,0.8411190566333265,-2.099206336787116])
        #pos[0] = (self.obj_pos[0] - 0.1)
        #pos[1] = (self.obj_pos[1] - 0.25) 
        #pos[2] += 0.1
        #for i in range(19):
        #   self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gvalue)

        #input("raw") 
#        pos = self.obj_pos#self.robot.getEndEffectorPos()#[self.obj_x-0.2,self.obj_y-0.3,self.obj_z+0.3]
#        orn = self.p.getQuaternionFromEuler([3.0111602527051042,0.8411190566333265,-2.099206336787116])
#        pos[0] = (self.obj_pos[0] - 0.1)
#        pos[1] = (self.obj_pos[1] - 0.2) 
#        pos[2] += 0.1
#        for i in range(19):
#           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gvalue)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        print("initialized")

    def get_success(self, seg=None):
        ID_SHOVEL = 22
        pos_shovel = self.p.getLinkState(self.robotId, ID_SHOVEL)[0]
        #if pos_shovel[2] < 0.36:
        #    return False
        #for id_obj in self.mlist:
        #    pos_obj = self.p.getBasePositionAndOrientation(id_obj)[0]
        #    if pos_obj[2] > pos_shovel[2]:
        #        return True
        return False
