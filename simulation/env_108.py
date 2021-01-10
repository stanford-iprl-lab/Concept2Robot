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
from math import sin,cos,acos
import matplotlib.pyplot as plt

import sys
sys.path.append('./Eval')
sys.path.append('./')

from env import Engine
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

def angleaxis2quaternion(angleaxis):
  angle = np.linalg.norm(angleaxis)
  axis = angleaxis / (angle + 0.00001)
  q0 = cos(angle/2)
  qx,qy,qz = axis * sin(angle/2)
  return np.array([qx,qy,qz,q0])

def quaternion2angleaxis(quater):
  angle = 2 * acos(quater[3])
  axis = quater[:3]/(sin(angle/2)+0.00001)
  angleaxis = axis * angle
  return np.array(angleaxis)


class Engine108(Engine):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine108,self).__init__(opti, wid=worker_id, p_id=p_id, maxSteps=maxSteps, taskId=taskId, n_dmps=n_dmps, cReward=cReward,robot_model=None)
        self.opti = opti

    
    def init_obj(self):
        self.obj_file = os.path.join(self.resources_dir,"urdf/objmodels/nut.urdf")
        self.obj_position = [0.3637 + 0.06, -0.06, 0.35]
        self.obj_scaling = 2
        self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2+0.2, -math.pi/2, -0.3])
        self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id

        self.box_file = os.path.join (self.resources_dir, "urdf/openbox/openbox.urdf")
        self.box_position = [0.27, 0.00, -0.30]
        self.box_scaling = 0.00044
        self.box_orientation = self.p.getQuaternionFromEuler ([0, math.pi, -math.pi/2])
        self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                      baseOrientation=self.box_orientation,
                                      globalScaling=self.box_scaling,useFixedBase=True)
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[38/255.,0.,128/255.0,1])
        self.p.changeDynamics(self.obj_id,-1,mass=2.0)

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id, self.obj_position, self.obj_orientation)


    def init_motion(self):
        self.data_q = np.load (os.path.join(self.robot_recordings_dir,"47-4/q.npy"))
        self.data_gripper = np.load (self.configs_dir + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])


    def init_grasp(self):
        self.box_position[2] = -.30
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.robot_recordings_dir, "47-4/q.npy"))
        glist = np.load( os.path.join(self.robot_recordings_dir, "47-4/gripper.npy"))
        num_q = len(qlist[0])
        self.fix_orn = np.load (os.path.join (self.configs_dir, 'init', 'orn.npy'))
        self.null_q = qlist[180]
        self.robot.setJointValue(qlist[40],glist[40])
        for i in range(40,180,1):
            glist[i] = min(120,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])

        pos = self.robot.getEndEffectorPos()
        pos[2] += 0.15
        orn = self.robot.getEndEffectorOrn()
        for i in range(109):
           self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=130)
#        time.sleep(3)
        self.start_pos = self.p.getLinkState (self.robotId, 7)[0]
        self.box_position[2] *= -1.0
        self.p.resetBasePositionAndOrientation(self.box_id,self.box_position,self.box_orientation)

    def step_dmp(self,action,f_w,coupling,reset):
        if reset:
          action = action.squeeze()
          self.start_pos = self.robot.getEndEffectorPos()
          self.start_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
          self.start_gripper_pos = self.robot.getGripperPos()
          self.start_status = np.array([self.start_pos[0],self.start_pos[1],self.start_pos[2],self.start_orn[0],self.start_orn[1],self.start_orn[2],0.0]).reshape((-1,))
          self.dmp.set_start(np.array(self.start_status)[:self.dmp.n_dmps])
          dmp_end_pos = [x+y for x,y in zip(self.start_status,action)]
          self.dmp.set_goal(dmp_end_pos)
          if f_w is not None:
            self.dmp.set_force(f_w)
          self.dmp.reset_state()
          #self.traj = self.dmp.gen_traj()
          self.actual_traj = []
          p1 = self.start_pos 
          p1 = np.array(p1)
          self.dmp.timestep = 0
          small_observation = self.step_within_dmp (coupling)
          #for idx, small_action in enumerate(self.traj):
          #  if idx < 7:
          #    for i in range(4):
          #       small_observation = self.step_within_dmp (small_action)
          #  else:
          #    small_observation = self.step_within_dmp (small_action)

          #self.actual_traj.append(tmp_pos)
          #self.a_traj = np.array(self.actual_traj)
          #p2 = self.robot.getEndEffectorPos()
          #p2 = np.array(p2)
          lenT = len(self.dmp.force[:,0])
          if self._wid == 0:
            fig = plt.figure(1)
          #  plt.plot(np.arange(0,lenT),self.traj[:,0],'--',color='r')
          #  plt.plot(np.arange(0,lenT),self.traj[:,1],'--',color='g')
          #  plt.plot(np.arange(0,lenT),self.traj[:,2],'--',color='b')
          #  plt.plot(np.arange(0,lenT),self.a_traj[:,0],color='red')
          #  plt.plot(np.arange(0,lenT),self.a_traj[:,1],color='green')
          #  plt.plot(np.arange(0,lenT),self.a_traj[:,2],color='blue')
            plt.plot(np.arange(0,lenT),self.dmp.force[:,0],color='red')
            plt.plot(np.arange(0,lenT),self.dmp.force[:,1],color='green')
            plt.plot(np.arange(0,lenT),self.dmp.force[:,2],color='blue')
            fig.canvas.draw()
            images = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3)
            cv2.imshow("Example",images)
            fig.canvas.figure.clf() 
            cv2.waitKey(1)
        else:
          small_observation = self.step_within_dmp(coupling)
        seg = None
        observation_next, seg = self.get_observation(segFlag=True)
        reward = 0
        done = False
        suc = False
        suc_info = self.get_success()
        if self.dmp.timestep >= self.dmp.timesteps:
          reward, done, suc = self.get_reward(seg)
          done = True
          self.success_flag = suc_info
        else:
          if np.sum(seg == 167772162) < 1:
            done = True
            self.success_flag = False
        return observation_next, reward, done, self.success_flag

    def get_success(self):
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
        #print(len (left_closet_info),len (right_closet_info),obj[0][0], box[1][0])
        if len (left_closet_info) > 0 and len (right_closet_info) > 0 and dist < 0.05:
          return True
        else:
          return False
