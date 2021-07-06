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

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


class Engine153(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=15, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine153,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti
        self._wid = worker_id
        self.max_steps = maxSteps 
        self.cReward = cReward
        self.robot.gripperMaxForce = 10000.0 

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
        self.obj_scaling = 0.9
        self.obj_id = self.p.loadURDF(os.path.join(self.urdf_dir,"obj_libs/bottles/b1/b1.urdf"),globalScaling=self.obj_scaling)

        self.box_file = os.path.join (self.urdf_dir, "obj_libs/cubes/c3/c3.urdf")
        self.box_position = [0.42, -0.02, 0.27]
        self.box_scaling = 1.5
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        self.p.changeVisualShape (self.box_id, -1, rgbaColor=[1.,0.,0.,1])

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,[0.3637 + 0.06, -0.05, 0.3515],[0, 0, -0.1494381, 0.9887711])
        self.p.resetBasePositionAndOrientation(self.box_id,[0.3337, 0.05, 0.2915],[0, 0, 0, 1])
        self.box_pos = np.array([0.3337,0.05,0.2915])
        self.prevDist = 0.23

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (os.path.join(self.robot_recordings_dir,"47-4/gripper.npy"))
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        qlist = np.load(os.path.join(self.robot_recordings_dir,'47-4/q.npy'))
        glist = np.load(os.path.join(self.robot_recordings_dir,'47-4/gripper.npy'))
        num_q = len(qlist[0])

        self.robot.setJointValue(qlist[0],glist[0])

        for i in range(10,len(qlist)):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
            #time.sleep(0.1)

        cur_joint = self.robot.getJointValue()
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = self.robot.getEndEffectorOrn()
        pos_diff = np.random.uniform(-0.1,0.1,size=(2,))
        cur_pos[:2] = cur_pos[:2] + pos_diff
        cur_pos[2] -= -0.01
        for i in range(19):
           self.robot.positionControl(cur_pos,cur_orn,null_pose=cur_joint,gripperPos=130)

        self.obj2_ori = np.array([0,0,0,1])
        trans = np.random.uniform(-0.1,0.1,size=(2,))
        self.obj2_posi = np.copy(self.box_pos)
        self.obj2_posi[:2] = self.box_pos[:2] + trans
        print("self.obj2",self.obj2_posi)
        self.p.resetBasePositionAndOrientation(self.box_id,self.obj2_posi,self.obj2_ori)


        self.fix_orn = self.p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,seg=None):
        obj = self.p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        gripper_pos = self.robot.getGripperTipPos()

        dist = np.linalg.norm(np.array(obj_center) - gripper_pos)
        obj_v = self.p.getBaseVelocity(self.obj_id)[0]
        obj_v_norm = np.linalg.norm(obj_v)


        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        box_AABB = self.p.getAABB(self.box_id)
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        if obj_v_norm > 1.0 and dist > 0.05 and len (left_closet_info)==0 and len (right_closet_info)==0 and obj_pos[0] > box_AABB[0][0] and obj_pos[0] < box_AABB[1][0] and obj_pos[1] > box_AABB[0][1] and obj_pos[1] < box_AABB[1][1]:
          return True
        else:
          return False
