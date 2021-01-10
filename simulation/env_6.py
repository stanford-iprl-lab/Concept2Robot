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
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('./')

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine6(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine6,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        #self.p.setPhysicsEngineParameter(enableConeFriction=1)
        #self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        #self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        #self.p.setPhysicsEngineParameter(numSolverIterations=20)
        #self.p.setPhysicsEngineParameter(numSubSteps=10)

        #self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        #self.p.setPhysicsEngineParameter(enableFileCaching=0)
        #self.p.setGravity(0,0,-9.81)


    def reset_new(self):
        #self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        #self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
        return self.get_observation()

    def init_obj(self):
        self.obj1_scaling = 1.2
        self.obj_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf"),globalScaling=self.obj1_scaling)
        self.obj2_scaling = 2.0
        self.obj2_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b6/b6.urdf"),globalScaling=self.obj2_scaling)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[0.3,0.3,0.9,1])
   
    def reset_obj(self):
        self.obj1_orn = self.p.getQuaternionFromEuler([math.pi/2.0,0,0])
        self.obj1_pos = np.array([0.31,0.03,0.3])

        r = R.from_quat(self.obj1_orn)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()
        HTrans[:3,3] = self.obj1_pos

        rotation_degree = np.random.uniform(-0.5,0.5)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)
    
        trans = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj1_pos[:2] = self.obj1_pos[:2] + trans

        self.obj1_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()

        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj1_pos,self.obj1_orn)
        self.p.resetBasePositionAndOrientation(self.obj2_id,[0.3796745705777075, -0.16140520662377353 + 0.05, 0.40416254394211476-0.03],[0, 0, -0.1494381, 0.9887711])
  
        obj_friction_ceof = 10.0
        self.p.changeDynamics(self.obj_id, -1, mass=1.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)


        obj2_friction_ceof = 10.0
        self.p.changeDynamics(self.obj2_id, -1, mass=1.9)
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
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"29-0/q.npy"))
        self.data_gripper = np.load(os.path.join(self.robot_recordings_dir,"29-0/gripper.npy"))#np.load (self.env_root + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "29-1/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "29-1/gripper.npy"))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[0],glist[0])
        for i in range(0,5,1):
            glist[i] = max(200,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
 
        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,seg=None):
        objid_pixel = np.sum(seg == self.obj_id)
        obj1_pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        obj2_pos = np.array(self.p.getBasePositionAndOrientation(self.obj2_id)[0])
        dist = np.linalg.norm(obj1_pos - obj2_pos)
        cl = self.p.getClosestPoints(bodyA=self.obj_id,bodyB=self.obj2_id,distance=100)
        contact = False
        if len(cl) > 0:
          if cl[0][8] < 0.04:
            contact = True
        if objid_pixel < 10 and contact:
          return True
        else:
          return False

