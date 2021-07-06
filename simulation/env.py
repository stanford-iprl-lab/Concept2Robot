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

import sys
sys.path.append('./Eval')
sys.path.append('./')

#### import deepTraj
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMP_DIR = os.path.join(BASE_DIR,'../deepTraj')
print("dmp in environmnet",DMP_DIR)
sys.path.insert(0,DMP_DIR)
from ddmp import DDMP  as DMP

try:
    from .utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
    from .robot import Robot
except Exception:
    from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code   
    from robot import Robot

#################################
import signal
import importlib
import torch
import torch.nn as nn

import sh
import re
import torch.nn.functional as F

np.set_printoptions(precision=4,suppress=True,linewidth=300)
sys.path.insert(0,"../external/something-something-v2-baseline.git")
sys.path.insert(0,"../classification/image")
sys.path.insert(0,"../../")


################################
from utils import *
import torchvision
from transforms_video import *

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


class Engine:
    def __init__(self, opti, wid, p_id, maxSteps=23, taskId=15, n_dmps=3, cReward=True, robot_model=None):
        # TODO: Make less hacky
        self.classifier = opti.classifier
        if self.classifier == 'video':
          print("printting video")
#          self.config = load_json_config("../../models/something-something-v2-baseline/configs/config_model1.json")
          self.config = load_json_config("../models/something-something-v2-baseline/configs/config_model1_left_right.json")
        elif self.classifier == 'image':
          # load configurations
          self.config = load_json_config("../classification/image/configs/config_resnet.json")
        elif self.classifier == 'tsm_video':
          self.config = load_json_config("../classification/video/configs/config_tsm_video.json")
        print("self.config",self.config)

        # setup device - CPU or GPU
        self.device = torch.device("cuda")
        self.device_ids = [0]
        print("> Using device: {}".format(self.device.type))
        print("> Active GPU ids:{}".format(self.device_ids))

        self._wid = wid
        self.opti = opti
        self.p = p_id
        self.init_rl ()
        self.taskId = taskId
        self.cReward = cReward
        self.n_dmps = n_dmps

        self.p.setPhysicsEngineParameter(enableConeFriction=1)
        self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        if self.taskId == int(49):
          self.p.setPhysicsEngineParameter(numSolverIterations=20)
          self.p.setPhysicsEngineParameter(numSubSteps=10)
        else:
          self.p.setPhysicsEngineParameter(numSolverIterations=40)
          self.p.setPhysicsEngineParameter(numSubSteps=40)


        self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)

        self.p.setTimeStep(1 / 100.0)
        self.p.setGravity(0,0,-9.81)

        #if self.opti.video_reward:
        #    self.eval = self.opti.load_video_pred

        self.dmp = DMP(opti,self.n_dmps) #self.opt.load_dmp
        assert (self.opti.video_reward)

        self.configs_dir = os.path.join(self.opti.project_dir, 'configs')
        self.resources_dir = os.path.join(self.opti.project_dir, 'resources')

        self.urdf_dir = os.path.join(self.resources_dir,'urdf')
        self.robot_recordings_dir = os.path.join(self.opti.project_dir, 'data', 'robot_recordings')
        #self.log_root = os.path.join(opti.project_root,'logs')
        #self.log_root = safe_path(self.log_root+'/td3_log/test{}'.format(self.test_id))
        self.sim_dir = os.path.join(self.opti.project_dir,'simulation')
        print("self.opti.project_dir",self.opti.project_dir)
        self.env_dir = self.sim_dir

        self.init_plane ()
        self.init_table ()

        if robot_model is None:
          self.robot = Robot(pybullet_api=self.p,opti=self.opti)
        else:
          self.robot = robot_model(pybullet_api=self.p,opti=self.opti)
        self.robotId = self.robot.robotId
        self.robotEndEffectorIndex = self.robot.endEffectorIndex
        self.init_obj ()

        self.view_matrix, self.proj_matrix = get_view (self.opti)
        self.q_home = np.array((0., -np.pi/6., 0., -5./6.*np.pi, 0., 2./3.*np.pi, 0.))
        self.w = self.opti.img_w
        self.h = self.opti.img_h

        self.epoch_suc = False
        self.epoch_done = False
        self.obs_list = []

        self.max_steps = maxSteps
        self.env_step = 0
        print("self.cRewards",self.cReward)
        if self.cReward:
          self.load_model()
        self.init_motion ()
        self.success_flag = False

    def load_model(self):
        if self.classifier == 'video':
            save_dir =  os.path.join("../models/something-something-v2-baseline/pretrained/model3D_1_left_right/")
            print("save_dir",save_dir)
        elif self.classifier == 'trn_video':
            save_dir =  "/juno/u/lins2/TRN-pytorch"
            print("save_dir",save_dir)
        elif self.classifier == 'image':
            save_dir =  os.path.join("../models/classification/image/model_resnet/")
        elif  self.classifier == 'tsm_video':
            save_dir =  os.path.join(self.config['output_dir'])

        signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

        # set column model
        if self.classifier == 'tsm_video':
            from classification.video.models.tsm import TSN
            model = TSN(self.config["num_classes"],
                        num_segments=8,  # depending on the model file we use
                        modality='RGB',
                        base_model='resnet50',  # depending on the model file we use
                        consensus_type='avg',
                        img_feature_dim=256,  # depending on the model file we use
                        pretrain=True,
                        is_shift=True,
                        shift_div=8,  # depending on the model file we use
                        shift_place='blockres',
                        non_local=False,
                        )
        elif self.classifier == "trn_video":
          sys.path.insert(0,'/juno/u/lins2/TRN-pytorch/')
          from models import TSN
          model = TSN(self.config['num_classes'], 
                            num_segments=1,
                            modality='RGB',
                            base_model='BNInception',
                            consensus_type='avg',
                         )
        else:
          file_name = self.config['conv_model']
          print("file_name",file_name)
          cnn_def = importlib.import_module("{}".format(file_name))

          # create model
          if self.classifier == 'video':
            from models.multi_column import MultiColumn
            model = MultiColumn(self.config["num_classes"], cnn_def.Model, int(self.config["column_units"]))
          elif self.classifier == 'image':
            model = cnn_def.Model(self.config["num_classes"])
          elif self.classifier == 'tsm_video':
            save_dir =  os.path.join(self.config['output_dir'])


        # multi GPU setting
        model = torch.nn.DataParallel(model, self.device_ids).to(self.device)

        # optinally resume from a checkpoint
        if  self.classifier == 'tsm_video':
          checkpoint_path = os.path.join(save_dir,'model_best.pth')
        elif self.classifier == 'tsn_video':
          checkpoint_path = os.path.join(save_dir, 'model_best.path.tar')
        else:
          checkpoint_path = os.path.join(save_dir,'model_best.pth.tar')

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("Loading checkpoint for worker id_",self._wid)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        checkpoint = torch.load(checkpoint_path)
        #start_epoch = checkpoint['epoch']
        #best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model
        self.prevDist = 0.0

    def getLinkInfo(self, object_id):
        numJoint = self.p.getNumJoints(object_id)
        LinkList = ['base']
        for jointIndex in range(numJoint):
            jointInfo = self.p.getJointInfo(object_id, jointIndex)
            print("joiniNFO", jointInfo)
            link_name = jointInfo[12]
            if link_name not in LinkList:
                LinkList.append(link_name)
        return LinkList

    def getNumLinks(self, object_id):
        return len(self.getLinkInfo(object_id))

    def getAABB(self, object_id):
        numLinks = self.getNumLinks(object_id)
        AABB_List = []
        for link_id in range(-1, numLinks - 1):
            AABB_List.append(self.p.getAABB(object_id, link_id))
        AABB_array = np.array(AABB_List)
        AABB_obj_min = np.min(AABB_array[:, 0, :], axis=0)
        AABB_obj_max = np.max(AABB_array[:, 1, :], axis=0)
        AABB_obj = np.array([AABB_obj_min, AABB_obj_max])
        return AABB_obj

    def reset_obs_list(self):
        self.epoch_suc =  False
        self.epoch_done = False
        self.obs_list = []

    def init_table(self):
        if 1:
          table_path = os.path.join(self.resources_dir,'urdf/table/table.urdf')
          self.table_id = self.p.loadURDF(table_path, [0.42,0,0],[0,0,math.pi*0.32,1],globalScaling=0.6)#,physicsClientId=self.physical_id)
          texture_path = os.path.join(self.resources_dir,'textures/table_textures/table_texture.jpg')
          self.table_textid = self.p.loadTexture (texture_path)
          self.p.changeVisualShape (self.table_id, -1, textureUniqueId=self.table_textid)
        #print(self.getAABB(self.table_id))
        #print(self.p.getAABB(self.table_id))
        #print(self.getNumLinks(self.table_id))
        #input("raw")

        else:
          max_z = 0.289
          min_z = 0.0
          max_y = 0.4556
          min_y = -max_y
          max_x = 0.7264
          min_x = 0.1136
          self.table_height = max_z
          self.table_width = max_y * 2.0
          self.table_depth = max_x - min_x
          self.table_v = self.p.createVisualShape(self.p.GEOM_BOX,
                                                halfExtents=[self.table_depth / 2.0, self.table_width / 2.0,
                                                             self.table_height / 2.0])
          self.table_c = self.p.createCollisionShape(self.p.GEOM_BOX,
                                                   halfExtents=[self.table_depth / 2.0, self.table_width / 2.0,
                                                                self.table_height / 2.0])
          mass = 0
          self.table_id = self.p.createMultiBody(mass, baseCollisionShapeIndex=self.table_c,
                                               baseVisualShapeIndex=self.table_v, basePosition=(self.table_depth / 2.0 + min_x, 0.0, self.table_height / 2.0))
          self.table_color = [128 / 255.0, 128 / 255.0, 128 / 255.0, 1.0]
          self.p.changeVisualShape(self.table_id, -1, rgbaColor=self.table_color)


    #def reset_table(self):
    #    self.p.changeVisualShape (self.table_id, -1, textureUniqueId=self.table_textid)

    def init_motion(self):
        # TODO: use json
        self.data_q = np.load(os.path.join(self.configs_dir, 'init', 'q.npy'))
        self.data_dq = np.load(os.path.join(self.configs_dir, 'init', 'dq.npy'))
        self.data_gripper = np.load(os.path.join(self.configs_dir, 'init', 'gripper.npy'))
       
    def init_plane(self):
        self.plane_id = self.p.loadURDF(os.path.join(self.resources_dir, 'urdf', 'plane.urdf'),
                                        [0.7, 0, 0], [0, 0, -math.pi * 0.02, 1], globalScaling=0.7)
        texture_path = os.path.join(self.resources_dir, 'textures', 'real_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        self.textid = self.p.loadTexture(texture_file)
        self.p.changeVisualShape (self.plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        self.p.changeVisualShape (self.plane_id, -1, textureUniqueId=self.textid)


    def save_video(self,img_info,i):
        img = img_info[2][:, :, :3]
        mask = (img_info[4] > 10000000)
        mask_id_label = [234881025, 301989889, 285212673, 268435457, 318767105, 335544321, 201326593, 218103809, 167772161]
        for item in mask_id_label:
            mask = mask * (img_info[4] != item)
        img = cv2.cvtColor (img, cv2.COLOR_RGB2BGR)
        img[mask] = [127, 151, 182]
        cv2.imwrite (os.path.join (self.output_path, '%06d.jpg' % (i)), img)

        try:
            img = cv2.imread (os.path.join (self.frames_path, '%06d.jpg' % (i + 1)))
            img[mask] = [127, 151, 182]
            cv2.imwrite (os.path.join (self.mask_frames_path, '%06d.jpg' % (i)), img)
        except:
            print('no video frame:{}'.format(i))

    def init_obj(self):
        if self.opti.object_id == 'bottle':
            self.obj_file = os.path.join(self.resources_dir, "urdf/objmodels/urdfs/bottle1.urdf")
            self.obj_position = [0.4, -0.15, 0.42]
            self.obj_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            self.obj_scaling = 1.4
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

        if self.opti.object_id == 'cup':
            self.obj_file = os.path.join(self.resources_dir,"urdf/objmodels/urdfs/cup.urdf")
            self.obj_position = [0.45, -0.18, 0.34]
            self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            self.obj_scaling = 0.11
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

        if self.opti.object_id == 'nut':
            self.obj_file = os.path.join(self.resources_dir,"urdf/objmodels/nut.urdf")
            self.obj_position = [0.4, -0.15, 0.34]
            self.obj_scaling = 2
            self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)
            self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.3,0.3,0.9,1])
        texture_path = os.path.join(self.resources_dir,'textures/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])


    def reset_obj(self):
        if self.opti.object_id == 'bottle':
            self.obj_position = [0.4, -0.15, 0.42]
            self.obj_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            self.obj_scaling = 1.4
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)#,physicsClientId=self.physical_id)

        if self.opti.object_id == 'cup':
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

        if self.opti.object_id == 'nut':
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

       
    def run(self):
        for i in range(self.data_q.shape[0]):
            jointPoses = self.data_q[i]
            for j in range(self.robotEndEffectorIndex):
                p.resetJointState(self.robotId, j, jointPoses[j], self.data_dq[i][j])

            gripper = self.data_gripper[i]
            self.gripperOpen = 1 - gripper / 255.0
            self.gripperPos = np.array (self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array (
                self.gripperLowerLimitList) * self.gripperOpen
            for j in range (6):
                index_ = self.activeGripperJointIndexList[j]
                p.resetJointState (self.robotId, index_, self.gripperPos[j], 0)

            img_info = self.p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=p.ER_TINY_RENDERER)
            self.save_video(img_info,i)
            self.p.stepSimulation()

    def get_traj(self):
        pos_traj, orn_traj = [], []
        for i in range (self.data_q.shape[0]):
            poses = self.data_q[i]
            for j in range (7):
                p.resetJointState (self.robotId, j, poses[j], self.data_dq[i][j])

            state = p.getLinkState (self.robotId, 7)
            pos = state[0]
            orn = state[1]
            pos_traj.append (pos)
            orn_traj.append (orn)

    def init_grasp(self):
        pos_traj = np.load(os.path.join(self.config_dir, 'init', 'pos.npy'))
        orn_traj = np.load(os.path.join(self.config_dir, 'init', 'orn.npy'))
        self.fix_orn = np.load(os.path.join(self.config_dir, 'init', 'orn.npy'))
 
        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = self.p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        start_id = 0
        init_traj = point2traj(points)
        start_id = self.move(init_traj,orn_traj,start_id)

        self.p.stepSimulation()

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            gripperPos = (grasp_t + 1.)/ float(grasp_stage_num) * 250.0 + 0.0
            self.robot.gripperControl(gripperPos)

            start_id += 1

        pos = p.getLinkState (self.robotId, 7)[0]
        left_traj = point2traj([pos, [pos[0], pos[1]+0.14, pos[2]+0.05]])
        start_id = self.move(left_traj, orn_traj,start_id)

        self.start_pos = p.getLinkState(self.robotId,7)[0]

    def move_up(self):
        # move in z-axis direction
        orn_traj = np.load(os.path.join(self.configs_dir, 'init', 'orn.npy'))
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1], pos[2] + 0.3]],delta=0.005)
        start_id = self.move (up_traj, orn_traj,0)

    def explore(self,traj):
        orn_traj = np.load ('orn.npy')
        start_id = self.move (traj, orn_traj,0)

    def move(self,pos_traj,orn_traj,start_id=0):
        for i in range(int(len(pos_traj))):
            pos = pos_traj[i]
            orn = orn_traj[i]
            self.robot.positionControl(pos=pos,orn=orn,null_pose=self.data_q[i])

            img_info = self.p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=self.p.ER_TINY_RENDERER)
            # self.save_video(img_info,start_id+i)
        return start_id+len(pos_traj)

    def init_rl(self):
        """
        self.target_pos = [0.3633281737186908, -0.23858468351424078, 0.670415682662147]
        # self.target_pos = [0.3633281737186908, -0.23858468351424078+0.1, 0.670415682662147-0.1]
        self.each_action_lim = self.opti.each_action_lim
        low = [-self.each_action_lim]*3
        high = [self.each_action_lim]*3
        self.action_space = {'low':low,'high':high}
        self.max_seq_num = 500
        self.min_dis_lim = self.opti.end_distance
        self.axis_limit = [eval(self.opti.axis_limit_x),eval(self.opti.axis_limit_y),
                           eval(self.opti.axis_limit_z)]
        """
        pass

    def init_dmp(self):
        """
        if self.opt.use_dmp and self.opt.dmp_imitation:
            trajectories = []
            for file in os.listdir (self.opt.actions_root):
                action_id = int (file.split ('-')[0])
                video_id = int (file.split('-')[1])
                if (action_id == self.opt.action_id) and (video_id==self.opt.video_id):
                    self.now_data_q = np.load (os.path.join (self.opt.actions_root, file, 'q.npy'))
                    pos_traj, orn_traj = [], []
                    for i in range (self.now_data_q.shape[0]):
                        poses = self.now_data_q[i]
                        for j in range (7):
                            p.resetJointState (self.robotId, j, poses[j], self.now_data_q[i][j])
                        state = p.getLinkState (self.robotId, 7)
                        pos = state[0]
                        orn = state[1]
                        pos_traj.append (pos)
                        orn_traj.append (orn)
                    trajectories.append (np.array (pos_traj))
            dmp_imitation_data = np.array (trajectories)
            print("dmp.imitate")
#            self.dmp.imitate (dmp_imitation_data)
        """
        pass

    def reset(self):
        self.seq_num = 0
        self.init_dmp()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
        self.reset_obs_list()
        self.real_traj_list = []
        self.obs_list = [] 
        observation = self.get_observation()      
        self.env_step = 0
        return observation


    def step(self,action=None,f_w=None,coupling=None,reset=True,test=False):
        self.env_step += 1
        return_value = self.step_dmp(action,f_w,coupling,reset,test)
        return return_value

    def tableDet(self):
        pos_gripper = self.robot.getEndEffectorPos()
        tableAABB = self.p.getAABB(self.table_id)
        if pos_gripper[0] > tableAABB[1][0] or pos_gripper[1] < tableAABB[0][1] or pos_gripper[1] > tableAABB[1][1]:
          return True
        else:
          return False


    def step_dmp(self,action,f_w,coupling,reset,test=False):
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
          self.dmp.rollout()
          self.dmp.reset_state()
          self.actual_traj = []
          p1 = self.start_pos
          p1 = np.array(p1)
          self.dmp.timestep = 0
          small_observation = self.step_within_dmp (coupling)
          lenT = len(self.dmp.force[:,0])
        else:
          small_observation = self.step_within_dmp(coupling)
        seg = None
        observation_next, seg = self.get_observation(segFlag=True)
        reward = 0
        done = False
        suc = False
        if self.dmp.timestep >= self.dmp.timesteps:
         if test:
          self.success_flag = self.get_success(seg)
          reward = self.get_reward(seg)
          self.termination_flag = True
         else:
          self.termination_flag = True
          if np.sum(seg == 167772162) < 1:
            self.success_flag = False
            reward = -0.1
          elif self.robot.colliDet():
            self.success_flag = False
            reward = -0.1
          elif self.tableDet():
            self.success_flag = False
            reward = -0.1
          elif self.taskColliDet():
            self.success_flag = False
            reward = -0.1
          else:
            self.success_flag = self.get_success(seg)
            reward = self.get_reward(seg)
        else:
         if test:
          self.success_flag = self.get_success(seg)
          #if self.success_flag:
          #  self.termination_flag = True
          reward = self.get_reward(seg)
          self.termination_flag = False
         else:
          if np.sum(seg == 167772162) < 1:
            self.success_flag = False
            reward = -0.1
            self.termination_flag = True
          elif self.robot.colliDet():
            self.success_flag = False
            reward = -0.1
            self.termination_flag = True
          elif self.tableDet():
            self.success_flag = False
            reward = -0.1
            self.termination_flag = True
          elif self.taskColliDet():
            self.success_flag = False
            reward = -0.1
            self.termination_flag = True
          else:
            self.success_flag = self.get_success(seg)
            if self.success_flag:
              self.termination_flag = True
              reward = self.get_reward(seg)
            else:
              self.termination_flag = False
              if self.classifier == 'image' and self.opti.only_coupling_term == True:
                reward = self.get_reward(seg)
        return observation_next, reward, self.termination_flag, self.success_flag


    def get_current_traj_pos(self):
        cur_pos = self.robot.getEndEffectorPos()
        cur_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
        cur_gripper_pos = self.robot.getGripperPos()
        return np.array([cur_pos[0],cur_pos[1],cur_pos[2],cur_orn[0],cur_orn[1],cur_orn[2],cur_gripper_pos])


    def step_within_dmp(self,coupling):
        action = self.dmp.step(coupling)[0]
        pos = np.array(action[:3])
        if len(action) > 3:
          orn = angleaxis2quaternion(action[3:6])
        else:
          orn = self.fix_orn[0]
        if len(action) == 7:
          if action[6] < 0:
            gripperPos = self.start_gripper_pos + int(action[6] * 255)# + 127.5
          else:
            gripperPos = None
        else:
          gripperPos = None
        self.robot.positionControl(pos,orn=orn,null_pose=self.data_q[0],gripperPos=gripperPos)
        observation_next = None
        self.real_traj_list.append(self.robotCurrentStatus())
        return observation_next


    def step_without_dmp(self,action):
        action = action.squeeze()
        pos = np.array(self.robot.getEndEffectorPos()) + np.array(action)
        self.robot.positionControl(pos,self.data_q[0])
        seg = None
        observation_next, seg = self.get_observation(segFlag=True)
        reward = 0
        done = False
        reward, done, suc = self.get_reward(seg)
        return observation_next, reward, done, suc

    def get_observation(self,segFlag=False):
        # get observation
        img_info = self.p.getCameraImage (width=self.w,
                                     height=self.h,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     shadow=-1,
                                     flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                     renderer=self.p.ER_TINY_RENDERER)
        img = img_info[2][:, :, :3]

        self.observation = img
        self.obs_list.append(img)

        if segFlag:
            seg = img_info[4] 
            return img, seg
        else:
            return img

    def rewardBaseline(self, imgs):
        transform_pre = ComposeMix([
                [Scale(int(1.4 * self.config['input_spatial_size'])), "img"],
                [torchvision.transforms.ToPILImage(), "img"],
                [torchvision.transforms.CenterCrop(self.config['input_spatial_size']), "img"],
                 ])

        # Transforms common to train and eval sets and applied after "pre" transforms
        transform_post = ComposeMix([
                [torchvision.transforms.ToTensor(), "img"],
                [torchvision.transforms.Normalize(
                           mean=[0.485, 0.456, 0.406],  # default values for imagenet
                           std=[0.229, 0.224, 0.225]), "img"]
                 ])

        if self.classifier == 'image':
            imgs = [imgs[0], imgs[-1]]

        imgs = transform_pre(imgs)
        imgs = transform_post(imgs)

        num_frames = len(imgs)
        if self.classifier == 'video':
            num_frames_necessary = 72
            if len(imgs) < 72:
                imgs.extend([imgs[-1]] *
                                (72 - len(imgs)))
            data = torch.stack(imgs)
            data = data.permute(1, 0, 2, 3)
        else:
            data = torch.stack(imgs)

        return data

    def robotCurrentStatus(self):
        self.curr_pos = self.robot.getEndEffectorPos()
        self.curr_orn = quaternion2angleaxis(self.robot.getEndEffectorOrn())
        self.curr_gripper_pos = self.robot.getGripperPos()
        self.curr_status = np.array([self.curr_pos[0],self.curr_pos[1],self.curr_pos[2],self.curr_orn[0],self.curr_orn[1],self.curr_orn[2],0.0]).reshape((-1,))
        return self.curr_status

    def get_video_reward(self,seg,taskId=None):
        if taskId is None:
          taskId = self.taskId
        obs_list = self.obs_list.copy()
        data = self.rewardBaseline(obs_list)
        input = data.unsqueeze(0)
        input_var = [input.to(self.device)]
        output = self.model(input_var,False)
        output = F.softmax(output,1)
        output = output.cpu().detach().numpy()
        output = np.squeeze(output)
        if taskId < len(output):
          reward = output[taskId] * 174.0 - 2.0
          reward = 1. / (1. + np.exp(-reward))
        else:
          reward = 0.0
        return reward
 
    def get_tsm_video_reward(self, taskId=None):
        if taskId is None:
          taskId = self.taskId
        obs_list = self.obs_list.copy()
        data = self.rewardBaseline(obs_list)
        #TODO: @Karen Hard coding some params here, to be replaced
        length = 3 #RGB modality
        this_test_segments = 8
        if data.size(0) < this_test_segments:
            return 0.0
        segments = data.size(0)//this_test_segments * this_test_segments
        data_in = data[-segments:, :,:,:] #take the most recent part. Timesteps be divisible by this_test_segments
        data_in = data_in.view(-1, length, data.size(2), data.size(3))
        data_in = data_in.view(-1, this_test_segments, length, data_in.size(2), data_in.size(3))
        output = self.model(data_in,False)
        output = F.softmax(output,1)
        output = output.cpu().detach().numpy().flatten()
        output = np.squeeze(output)
        print("beflore",output.shape)
        reward = output[taskId] * 174.0 - 2.0
        reward = 1. / (1. + np.exp(-reward))
        return reward
 
    def taskColliDet(self): 
        return False

    def get_reward (self,seg=None):
        if self.cReward:
          if self.classifier in ('video', 'image'):
            return self.get_video_reward(seg)
          elif self.classifier == 'tsm_video':
            return self.get_tsm_video_reward()
        else:
            return float(self.get_success(seg))

    def get_success(self, seg):
        return False

    def get_handcraft_reward(self,seg=None):
        return 1.0, False, None
