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

class Robot:
    def __init__(self,pybullet_api,start_pos=[0.4,0.3,0.4],urdf_path=None,opti=None):
       self.p = pybullet_api

       self.gripperMaxForce = 100.0
       self.armMaxForce = 100.0
       self.endEffectorIndex = 7
       self.start_pos = start_pos

       #lower limits for null space
       self.ll=[-2.9671, -1.8326 ,-2.9671, -3.1416, -2.9671, -0.0873, -2.9671, -0.0001, -0.0001, -0.0001, 0.0, 0.0, -3.14, -3.14, 0.0, 0.0, 0.0, 0.0, -0.0001, -0.0001]
       #upper limits for null space
       self.ul=[2.9671, 1.8326 ,-2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0001, 0.0001, 0.0001, 0.81, 0.81, 3.14, 3.14, 0.8757, 0.8757, -0.8, -0.8, 0.0001, 0.0001]
       #joint ranges for null space
       self.jr=[(u-l) for (u,l) in zip(self.ul,self.ll)]

       # restposes for null space
       self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
       # joint damping coefficents
       self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

       self.num_controlled_joints = 7
       self.controlled_joints = [0, 1, 2, 3, 4, 5, 6]

       self.activeGripperJointIndexList = [10, 12, 14, 16, 19, 20]

       self.gripper_left_tip_index = 13
       self.gripper_right_tip_index = 17

#       model_path = "../../resources/urdf/panda_robotiq_peg.urdf"
       model_path = os.path.join("/juno/u/lins2/","ConceptManipulation/simulation/urdf","panda_robotiq_peg.urdf")
       print("model_path in urdf",model_path)

       self.robotId = self.p.loadURDF(model_path, [0, 0, 0],useFixedBase=True,flags=self.p.URDF_USE_SELF_COLLISION and self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT) 
       self.p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0], [0, 0, 0, 1])

       self.targetVelocities = [0] * self.num_controlled_joints
       self.positionGains = [0.03] * self.num_controlled_joints
       self.velocityGains = [1] * self.num_controlled_joints

       self.numJoint = self.p.getNumJoints(self.robotId)

       self.gripperLowerLimitList = []
       self.gripperUpperLimitList = []
       for jointIndex in range(self.numJoint):
            jointInfo = self.p.getJointInfo(self.robotId,jointIndex)
            print(self.p.getJointInfo(self.robotId,jointIndex), self.p.getJointInfo(self.robotId,jointIndex)[-1])
            if jointIndex in self.activeGripperJointIndexList:
                self.gripperLowerLimitList.append(jointInfo[8])
                self.gripperUpperLimitList.append(jointInfo[9])

#       print("19",self.p.getLinkState(self.robotId,20))
#       print("18AABB",self.p.getAABB(self.robotId,18))
#       print("18AABB",self.p.getAABB(self.robotId,20))
       print(self.p.getBodyInfo(self.robotId))
 
       #print("Peg",self.getPegPos())
 
    def reset(self): 
        ####### Set Dynamic Parameters for the gripper pad######
        friction_ceof = 1000.0
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, lateralFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, rollingFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

        self.p.changeDynamics(self.robotId, self.gripper_right_tip_index, lateralFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_right_tip_index, rollingFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

 
    def jointPositionControl(self,q_list,gripper=None,maxVelocity=None):
        q_list = q_list.tolist()
        if gripper is None:
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.controlled_joints,controlMode=self.p.POSITION_CONTROL,targetPositions=q_list)
        else:
            self.gripperOpen = 1 - gripper/255.0
            self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
            self.gripperPos = self.gripperPos.tolist()
            armForce = [self.armMaxForce] * len(self.controlled_joints)
            gripperForce = [self.gripperMaxForce] * len(self.activeGripperJointIndexList)
            
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.controlled_joints,controlMode=self.p.POSITION_CONTROL,targetPositions=q_list,forces=armForce)
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.activeGripperJointIndexList,controlMode=self.p.POSITION_CONTROL,targetPositions=self.gripperPos,forces=gripperForce)
        self.p.stepSimulation()

    def setJointValue(self,q,gripper):
        for j in range(len(self.controlled_joints)):
            self.p.resetJointState(self.robotId,j,q[j],0.0)
        self.gripperOpen = 1 - gripper/255.0
        self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
        for j in range(6):
            index_ = self.activeGripperJointIndexList[j]
            self.p.resetJointState(self.robotId,index_,self.gripperPos[j],0.0)

    def getJointValue(self):
        for j in range(len(self.controlled_joints)):
            self.p.getJointState(self.robotId,j,q[j],0.0)
    
    def getEndEffectorPos(self):
        return np.array(self.p.getLinkState(self.robotId, self.endEffectorIndex)[0])

    def getEndEffectorOrn(self):
        return np.array(self.p.getLinkState(self.robotId, self.endEffectorIndex)[1])

    def getEndEffectorVel(self):
        return self.p.getLinkState(self.robotId, self.endEffectorIndex)[6]

    def getGripperTipPos(self):
        left_tip_pos = self.p.getLinkState(self.robotId, self.gripper_left_tip_index)[0]
        right_tip_pos = self.p.getLinkState(self.robotId, self.gripper_right_tip_index)[0]
        gripper_tip_pos = 0.5 * np.array(left_tip_pos) + 0.5 * np.array(right_tip_pos)
        return gripper_tip_pos
   
    def operationSpacePositionControl(self,pos,orn=None,null_pose=None,gripperPos=None):
        orn = self.p.getQuaternionFromEuler([math.pi,0,0])
        if null_pose is None and orn is None:
            jointPoses = self.p.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr)[:self.num_controlled_joints]

        elif null_pose is None and orn is not None:
            jointPoses = self.p.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos, orn,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr)[:self.num_controlled_joints]

        elif null_pose is not None and orn is None:
            jointPoses = self.p.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr,
                                                      restPoses=null_pose)[:self.num_controlled_joints]

        else:
            jointPoses = self.p.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos, orn,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr,
                                                      restPoses=null_pose)[:self.num_controlled_joints]

        #print("jointPoses",jointPoses)
        if gripperPos is None:
          for i in range(5): 
            self.p.setJointMotorControlArray(bodyIndex=self.robotId,
                                        jointIndices=self.controlled_joints,
                                        controlMode=self.p.POSITION_CONTROL,
                                        targetPositions=jointPoses,
                                        targetVelocities=self.targetVelocities,
                                        forces=[self.armMaxForce] * self.num_controlled_joints)
        else:
            self.gripperOpen = 0.0#1.0 - gripperPos/255.0
            self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
            self.gripperPos = self.gripperPos.tolist()
            gripperForce = [self.gripperMaxForce] * len(self.activeGripperJointIndexList)
            jointPoses = np.array(jointPoses).tolist()
            for i in range(5):
              self.p.setJointMotorControlArray(bodyIndex=self.robotId,
                                        jointIndices=self.controlled_joints + self.activeGripperJointIndexList,
                                        controlMode=self.p.POSITION_CONTROL,
                                        targetPositions=jointPoses + self.gripperPos,
                                        targetVelocities=self.targetVelocities + [0.0] * len(self.activeGripperJointIndexList),
                                        forces=[self.armMaxForce] * self.num_controlled_joints + gripperForce)

        self.p.stepSimulation()

    def getGripperPos(self):
      jointInfo = self.p.getJointState(self.robotId,self.activeGripperJointIndexList[-1])
      angle = jointInfo[0]
      angle = (angle - self.gripperLowerLimitList[-1]) / (self.gripperUpperLimitList[-1]-self.gripperLowerLimitList[-1]) * 255.0
      return angle
    
    def getPegPos(self):
      PegAABB = self.p.getAABB(self.robotId,18)
      minZ = PegAABB[0][2]
      centerX = ( PegAABB[0][0] + PegAABB[1][0] ) * 0.5
      centerY = ( PegAABB[0][1] + PegAABB[1][1] ) * 0.5
      minZ = PegAABB[0][2]
      return np.array([centerX,centerY,minZ])
 
    def gripperControl(self,gripperPos):
        self.gripperOpen = 1.0 - gripperPos/255.0 
        self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
        self.gripperPos = self.gripperPos.tolist()
        #print("sef.gripperPos",self.gripperPos)
        gripperForce = [self.gripperMaxForce] * len(self.activeGripperJointIndexList)
        self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.activeGripperJointIndexList,controlMode=self.p.POSITION_CONTROL,targetPositions=self.gripperPos,forces=gripperForce)
        self.p.stepSimulation()

    def colliDet(self):
      for x in [10,11,12,13,14,15,16,17,18]:
        for y in [0,1,2,3,4,5,6]:
          c = self.p.getContactPoints(bodyA=self.robotId,bodyB=self.robotId,linkIndexA=x,linkIndexB=y)
          cl = self.p.getClosestPoints(bodyA=self.robotId,bodyB=self.robotId,distance=100,linkIndexA=x,linkIndexB=y)
#          print(cl)
          #print(c)
          if len(cl) > 0:
            if cl[0][8] < 0.02:
#              print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% True")
#              time.sleep(3)
              return True
      return False

