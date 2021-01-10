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

class Engine111(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine111,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti 
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

    def reset_new(self):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep (1/250.)
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
        self.obj_scaling = 1.2
        self.obj_pos = [0.3637 + 0.06, -0.05, 0.33]
        self.obj_ori = self.p.getQuaternionFromEuler([0.3,math.pi/2.0,math.pi/2.0])
        self.obj_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b8/b8.urdf"), self.obj_pos, self.obj_ori,globalScaling=self.obj_scaling)

        self.obj2_scaling = 1.0
        self.obj2_pos = [0.29,0.1,0.31]
        self.obj2_ori = self.p.getQuaternionFromEuler([math.pi/10.0,0,0.])
        self.obj2_id = self.p.loadURDF(os.path.join(self.resources_dir,"urdf/obj_libs/bevels/b1/b1.urdf"),self.obj2_pos,self.obj2_ori,globalScaling=self.obj2_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,0.,0.,1])
   
    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_pos,self.obj_ori)
  
        obj_friction_ceof = 0.3
        self.p.changeDynamics(self.obj_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)


        obj2_friction_ceof = 0.4
        self.p.changeDynamics(self.obj2_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj2_id, -1, lateralFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, rollingFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, spinningFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, linearDamping=20.0)
        self.p.changeDynamics(self.obj2_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj2_id, -1, contactStiffness=1.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load(os.path.join(self.configs_dir, 'init/gripper.npy'))
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
 
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,len(qlist),1):
            glist[i] = min(100,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
 
        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
