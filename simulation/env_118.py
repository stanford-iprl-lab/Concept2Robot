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

import matplotlib.pyplot as plt 

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine118(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine118,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.robot.gripperMaxForce = 10000.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        self.p.setPhysicsEngineParameter(numSolverIterations=20)
        self.p.setPhysicsEngineParameter(numSubSteps=10)

        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setGravity(0,0,-9.81)

    def reset_new(self):
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
        return self.get_observation()

    def init_obj(self):
        self.obj_id = self.p.loadURDF( os.path.join(self.resources_dir, "urdf/obj_libs/bottles/b1/b1.urdf"))
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1]) 
        self.box_pos_default = [0.6,0.0,-1.4]
        self.box_ori_default = self.p.getQuaternionFromEuler([0.0,0.0,math.pi/2.0])
        self.box_scaling = 1.5
        self.box_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/drawers/d7/d7.urdf"),self.box_pos_default,self.box_ori_default,globalScaling=self.box_scaling,useFixedBase=True)

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,[0.3637 + 0.06, -0.05, 0.34],[0, 0, -0.1494381, 0.9887711])
        obj_friction_ceof = 20000.0
        self.p.changeDynamics(self.obj_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

        self.box_pos = np.array(self.box_pos_default)
        self.box_ori = np.array(self.box_ori_default)

        r = R.from_quat(self.box_ori_default)
        HTrans = np.zeros((4,4))
        HTrans[:3,:3] = r.as_dcm()
        HTrans[:3,3] = self.box_pos

        rotation_degree = np.random.uniform(-0.8,0.8)
        addRot = R.from_rotvec(rotation_degree * np.array([0,0,1]))
        addHTrans = np.zeros((4,4))
        addHTrans[:3,:3] = addRot.as_dcm()
        NewHTrans = addHTrans.dot(HTrans)

        self.box_pos = NewHTrans[:3,3]
        transl = np.random.uniform(-0.05,0.05,size=(2,))
        self.box_pos[0] += transl[0]
        self.box_pos[1] += transl[1]

        self.box_ori = R.from_dcm(NewHTrans[:3,:3]).as_quat()
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_pos,self.box_ori)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        #self.box_pos_default[2] = 0.4
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_pos_default,self.box_ori_default)
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        gripper_v = 130
        self.robot.setJointValue(qlist[40],glist[40])
        self.p.resetBasePositionAndOrientation(self.obj_id,[0.3637 + 0.06, -0.05, 0.34],[0, 0, -0.1494381, 0.9887711])
        for i in range(40,180,1):
            glist[i] = min(gripper_v,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])


        pos = [0.28,0.0,0.35]
        orn = self.robot.getEndEffectorOrn()
        self.null_q = self.robot.getJointValue()
        for i in range(30):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gripper_v)

        transl = np.random.uniform(-0.05,0.05,size=(2,))
        pos = self.robot.getEndEffectorPos()
        pos[:2] += transl
        for i in range(30):
           self.robot.positionControl(pos,orn,null_pose=self.null_q,gripperPos=gripper_v)

        pos = self.p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+0.3]])
 
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
  
        self.box_pos[2] = 0.4
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_pos,self.box_ori)


    def get_success(self,seg=None):
        cameraEyePosition = np.array([0.4,0.0,2.0])#self.box_pos#viewMatrix[:3,3] 
        #cameraEyePosition[2] += 1.0 
        cameraUpVector = np.array([1,0,0])#viewMatrix[:3,1] * -1.0
        cameraTargetPosition = np.array([0.4,0.0,0.0])#viewMatrix[:3,3] - np.array()#+ viewMatrix[:3,2] * 0.001
        #cameraTargetPosition[2] = 0.0
        self._top_view_matrix = self.p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)

        img = self.p.getCameraImage (width=320,
                                         height=240,
                                         viewMatrix=self._top_view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER)

        pos1 = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        pos1 = np.array(pos1)
        dist = np.linalg.norm(pos1[:2] - np.array(self.box_pos)[:2])
        #plt.figure(0)
        #plt.imshow(img[2])
        #plt.figure(1)
        #plt.imshow(img[4])
        #print("dist",dist,np.sum(img[4] == self.obj_id))
        #plt.show()
 
        if dist < 0.3 and np.sum(img[4] == self.obj_id) < 10:
          return  True
        else:
          return False
