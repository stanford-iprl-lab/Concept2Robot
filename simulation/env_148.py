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

class Engine148(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine148,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
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
        self.obj_id = self.p.loadURDF( os.path.join(self.urdf_dir, "obj_libs/bottles/b1/b1.urdf"))
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.,0.2,0.8,1]) 

        self.box_file = os.path.join (self.urdf_dir, "openbox4/openbox.urdf")
        print("self.box_file",self.box_file)
        self.box_position = [0.40, -0.05, 0.29]
        self.box_scaling = 5.0
        self.box_orientation = self.p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)

        for i in range(5):
          self.p.changeVisualShape (self.obj_id, i, rgbaColor=[1.,0.,0.,1]) 

        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics (self.obj_id, -1, mass=20.0)
        self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, angularDamping=20.0)
        self.p.changeDynamics (self.obj_id, -1, contactStiffness=0.1, contactDamping=0.1)



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

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))

        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+0.3]])
 
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]

    def get_success(self,seg=None):
        obj_pos = self.p.getBasePositionAndOrientation(self.obj_id)[0]
        # check whether the object is still in the gripper
        left_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_left_tip_index, -1)
        right_closet_info = self.p.getContactPoints (self.robotId, self.obj_id, self.robot.gripper_right_tip_index, -1)
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and obj_pos[2] > 0.4:
          return True
        else:
          return False
                            
