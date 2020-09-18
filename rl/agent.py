import copy
import numpy as np
import os
import sys
import time
np.set_printoptions(precision=4,suppress=False)

import importlib
import glob
import imageio
import math
from tensorboardX import SummaryWriter
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMP_DIR = os.path.join(BASE_DIR,'../deepTraj')
sys.path.insert(0,DMP_DIR)
from ddmp import DDMP  as DMP

HOME_DIR = "/juno/u/lins2"
sys.path.insert(0, "../../simulation")
sys.path.insert(0,os.path.join(HOME_DIR,'bullet3/build_cmake/examples/pybullet'))
import pybullet

import bullet_client as bc
from config import opti

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

A_DIM = 7 * 50
TAU = 0.01

from scipy.special import softmax

#####
from actor import Actor
from critic import Critic
from master import Master
from grac import ASC

class Params:
  def identity(x):
    x
 
  def __init__(self,TDI=0,max_ep_demon=10000,recordGif=False,videoReward=True,only_coupling_term=False,only_force_term=False,use_cem=False,max_action=0.5,mem_c_capacity=10000,coupling_term=False,coupling_dim=7,traj_timesteps=49,restore_dir=None,test=False,method="ASC",max_ep_test=100,force_term=False,rotation_max_action=0.0,gui=False,max_ep=50 * 1000,mem_capacity=10000,a_coupling_lr=1e-5,c_coupling_lr=1e-5,traj_lr=1e-5,a_lr=1e-5,c_lr=5e-5,gamma=0.9,batch_size=32,update_step=20,entropy=0.0,action_penalty=0.01,a_dim=7,state_dim=120 * 160 *3,task_dim=1024,exp_name='v1_force_term',task_id=5,explore_var=0.5,explore_decay=0.9999,start_learning=2000,start_learning_coupling=5000):
    self.TDI = TDI
    self.use_cem = use_cem
    self.max_ep = max_ep
    self.videoReward = videoReward
    self.max_ep_demon = max_ep_demon
    self.recordGif = recordGif
    self.max_action = max_action
    self.mem_c_capacity = mem_c_capacity
    self.start_learning_coupling = start_learning_coupling
    self.max_ep_test = max_ep_test
    self.force_term = force_term
    self.coupling_term = coupling_term
    self.only_coupling_term = only_coupling_term
    if self.only_coupling_term:
      self.coupling_term = True
      self.only_force_term = False
      self.force_term = False
    self.only_force_term = only_force_term
    if self.only_force_term:
      self.force_term = True
    self.mem_capacity = mem_capacity
    self.coupling_dim = coupling_dim
    self.a_lr = a_lr
    self.c_lr = c_lr
    self.traj_lr = traj_lr
    self.a_coupling_lr = a_coupling_lr
    self.c_coupling_lr = c_coupling_lr
    self.gamma = gamma
    self.batch_size = batch_size
    self.update_step = update_step
    self.entropy = entropy
    self.action_penalty = action_penalty
    self.a_dim = a_dim
    self.state_dim = state_dim
    self.task_dim = task_dim
    self.exp_name = exp_name
    self.task_id = task_id
    self.traj_timesteps = traj_timesteps
    self.explore_var = explore_var
    self.explore_decay = explore_decay
    self.start_learning = start_learning
    self.gui = gui
    self.rotation_max_action = rotation_max_action
    self.method = method
    if self.TDI == 2 or self.TDI ==4:
      self.file_name = '{}/{}'.format("imitation",datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
      self.file_name = '{}_{}/{}_{}'.format(str(self.task_id),str(self.exp_name),str(self.method),datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("file_anme",self.file_name)
    print("test",test,"self.TDI",self.TDI)
    if not test or (test and self.TDI == 2) or (test and self.TDI == 4):
      self.writer = SummaryWriter(logdir=os.path.join('./save_logger',self.file_name))
      self.model_dir = os.path.join('./save_model',self.file_name)
      if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
    else:
      self.model_dir = restore_dir
      self.writer = None

transforms = transforms.Compose([
  transforms.ToPILImage(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  ])


def set_init(layers):
  for layer in layers:
    nn.init.normal_(layer.weight, mean=0., std=0.05)
    nn.init.constant_(layer.bias, 0.)

class Enver(object):
  def __init__(self,opti,task_id,params):
    self.params = params
    self.TaskId = task_id
    self.wid = task_id
    self.p_id = bc.BulletClient(connection_mode=pybullet.GUI)
    time_sleep = np.random.uniform(0,100)
    env_module = importlib.import_module("env_{}".format(self.TaskId))
    RobotEnv = getattr(env_module, "Engine{}".format(self.TaskId))
    self.env = RobotEnv(worker_id=self.wid, opti=opti, cReward=self.params.videoReward, p_id=self.p_id, taskId=self.TaskId, n_dmps=7)


class Worker(object):
  def __init__(self,wid,opti,params):
    self.wid = wid
    self.opti = opti
    self.params = params
    print("self.parms.force_term",self.params.force_term)
    self.TaskId = self.params.task_id
    print("worker id %d" % self.wid,"task id",self.TaskId)
    if opti.gui:
      self.p_id = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self.p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
    self.agent = ASC(self.params)
    time_sleep = np.random.uniform(0,100)
    print("time_sleep",time_sleep)
    
    env_module = importlib.import_module("env_{}".format(self.TaskId))
    RobotEnv = getattr(env_module, "Engine{}".format(self.TaskId))
    self.env = RobotEnv(worker_id=self.wid, opti=opti, cReward=self.params.videoReward, p_id=self.p_id, taskId=self.TaskId, n_dmps=7)
    self.task_vec = np.load("../../Languages/"+str(self.TaskId)+'.npy')

    self.env_list = {}
    self.p_id_list = {}

  def train(self,restore_step=0,restore_path=None):
    var = self.params.explore_var  # control exploratin
    episode_check = 0
    suc_check = 0
    step_check = 0
    reward_check = 0.0

    if restore_path is not None:
      self.agent.restore(restore_step, restore_path)

    total_step = 0 + restore_step
    if total_step > 0:
      var = var * self.params.explore_decay ** (total_step)
    print("total_step",total_step,"var",var)

    for ep_ in range(1,self.params.max_ep):
      observation = self.env.reset()

      observation = np.reshape(observation,(-1,))

      reset_flag = True
      initial_pose = self.env.robotCurrentStatus()
      while True:
        # Added exploration noise
        if total_step < self.params.start_learning:
           action_ = np.random.uniform(-0.5,0.5,size=(A_DIM))
        else:
           if not self.params.force_term:
             action_ = np.zeros((50 * 7,))
             action_[:7] = self.agent.choose_action(observation, self.task_vec)
             print("HEREHEH", total_step, self.params.force_term)
           else:
             if self.params.only_force_term:
               action_ = np.zeros((50 * 7,))
               action_[7:] = self.agent.choose_action(observation, self.task_vec)
             else:
               action_ = self.agent.choose_action(observation, self.task_vec)

        action_[3:6] = (action_[3:6] * np.pi * 2.0)

        var = max(var, 0.1)
        action =  np.random.normal(action_, var)
        self.params.writer.add_scalar("train/var",var,ep_)
        action[:3] = action[:3].clip(-0.5,0.5)
        action[3:6] = (action[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)

        action_7 = np.zeros((7,))
        action_7[:7] = action[:7]

        f_7 = np.zeros((self.opti.traj_timesteps,7))

        if self.params.force_term:
          f_a_3 = action[7:].reshape((self.params.a_dim,self.opti.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          print("weigths",weights.shape)
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          observation_next, reward, done, suc = self.env.step(action_7, f_7, None, reset_flag)
        else:
          observation_next, reward, done, suc = self.env.step(action_7, None, None, reset_flag)

        reset_flag = False
        observation_c = np.copy(observation_next).reshape((-1,))

        while not done:
          coupling_ = None
          observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, coupling_, reset_flag)
          if not done:
            reward = 0

        if not self.params.videoReward:
          reward = float(suc) * 3.0

        if not self.params.force_term:
          action_penalty = np.sum(np.abs(action[:self.params.a_dim]))/float(self.params.a_dim) * self.params.action_penalty
        else:
          action_penalty = np.sum(np.abs(action[:self.params.a_dim]))/float(self.params.a_dim) * self.params.action_penalty + np.sum(np.abs(action[self.params.a_dim:]))/float(self.params.a_dim * self.params.traj_timesteps) * self.params.action_penalty * 0.1

        reward = reward - action_penalty #-  #np.linalg.norm(self.env.robotCurrentStatus() - self.env.dmp.goal[:3]) * 0.5

        observation_next = np.reshape(observation_next,(-1,))
        print("taskid",self.TaskId,"exp_name",self.params.exp_name, "ep_",ep_," action",action[:7],"action_",action_[:7],"reward",reward,"suc",suc,"dmp dist",np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]), self.env.robotCurrentStatus()[:3],self.env.dmp.goal[:3])

        while len(self.env.real_traj_list) < self.env.dmp.timesteps:
            self.env.real_traj_list.append(self.env.real_traj_list[-1])
        print("self.env.real_tral_list",len(self.env.real_traj_list))
        GT_traj = np.array(self.env.real_traj_list)
        GT_traj = GT_traj.reshape((-1,))

        step_check += 1
        reward_check += reward
        self.params.writer.add_scalar("train/reward",reward,ep_)
        self.agent.store_transition(initial_pose, GT_traj, observation , action, reward, done, self.task_vec)

        if ep_ > self.params.start_learning:
          var = var*self.params.explore_decay    # decay the action randomness
          if not self.params.only_coupling_term:
            a = self.agent.learn()
            print("starting to learn")

        observation  = observation_next
        total_step += 1

        if ep_ % 1000 == 0:
          current_performance = float(suc_check)/(episode_check+0.0001)
          print("suc rate %f reward %f var%f" % (current_performance, float(reward_check)/(episode_check+0.0001), var))
          print("saving models at step%d"  % ep_)
          self.agent.save_model(ep_)
          self.test(ep_)

        if done:
          self.params.writer.add_scalar("train/suc",suc,ep_)
          suc_check += suc
          episode_check += 1
          break

  def test(self,eval_step=0,restore_path=None):
    total_suc = 0

    print("testing!")
    if restore_path is not None:
      self.agent.restore(eval_step,restore_path)
      print("testing from restoring sth",restore_path)
      
    for ep_ in range(self.params.max_ep_test):
      observation = self.env.reset()
      observation = np.reshape(observation,(-1,))
      reset_flag = True

      tt = 0
      while True:
        action = np.zeros((50 * 7,))

        # Added exploration noise
        action_ = self.agent.choose_action(observation, self.task_vec)
 
        ### clip by the max action
        if self.params.only_force_term:
          action[7:] = action_
        else:
          if self.params.force_term:
            action = action_
          else:
            action[:7] = action_
        
        action = action.clip(-0.5,0.5)
        action[3:6] = (action[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)

        action_7 = np.zeros((7,))
        action_7[:7] = action[:7]

        f_7 = np.zeros((self.opti.traj_timesteps,7))

        if self.params.only_coupling_term:
          action[:7] = 0
          action_7 = np.zeros((7,))

        if self.params.force_term:
          f_a_3 = action[7:].reshape((self.params.a_dim,self.opti.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          observation_next, reward, done, suc = self.env.step(action_7, f_7, None, reset_flag)
        else:
          observation_next, reward, done, suc = self.env.step(action_7, None, None, reset_flag)

        reset_flag = False
        observation_c = np.copy(observation_next).reshape((-1,))
  
        while not done:
          coupling_ = np.zeros((7,))
          observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, coupling_, reset_flag)

        print("Tesing","taskid",self.TaskId,"ep_",ep_," action",action[:7],"action_",action_[:7],"reward",reward,"suc",suc,"dmp dist",np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]), self.env.robotCurrentStatus()[:3],self.env.dmp.goal[:3])

        observation  = observation_next

        if done:
          total_suc += float(suc)
          recordGif = True#self.params.recordGif
          if recordGif:
            classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
            recordGif_dir = os.path.join('./gif_dir',str(self.params.task_id))
            if not os.path.exists(recordGif_dir):
              os.makedirs(recordGif_dir)
            imageio.mimsave(os.path.join(recordGif_dir,str(self.params.task_id)+'_'+str(ep_)+'.gif'),self.env.obs_list)
          break

    perf = total_suc / float(self.params.max_ep_test)
    print("success performance",perf)

    if self.params.writer is not None:
      self.params.writer.add_scalar("test/suc",perf,eval_step)


  def demonstrate(self,eval_step=0,restore_path=None):
    total_suc = 0

    if restore_path is not None:
      self.agent.restore(eval_step,restore_path)
      print("demonstrating from restoring sth",restore_path)

    for ep_ in range(2000):
      observation = self.env.reset()
      observation = np.reshape(observation,(-1,))
      observation_init = np.copy(observation)

      reset_flag = True
      initial_pose = self.env.robotCurrentStatus()

      while True:
        action = np.zeros((50 * 7,))

        # Added exploration noise
        action_ = self.agent.choose_action(observation, self.task_vec)
        if self.params.force_term:
          action_origin = np.copy(action_)
          if self.params.rotation_max_action < 1e-6:
            action_force = action_origin[7:]
            action_tmp = action_force.reshape((self.params.a_dim,self.opti.traj_timesteps)).transpose()
            action_tmp[:,3:6] = 0
            action_origin[3:6] = 0.0
        else:
          action_origin = np.zeros((50*7,))
          action_origin[:7] = action_
          if self.params.rotation_max_action < 1e-6:
            action_origin[3:6] = 0.0

 
        ### clip by the max action
        if self.params.force_term:
          action = action_
        else:
          action[:7] = action_
        
        action = action.clip(-0.5,0.5)
        action[3:6] = (action[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)

        action_7 = np.zeros((7,))
        action_7[:7] = action[:7]

        f_7 = np.zeros((self.opti.traj_timesteps,7))

        if self.params.only_coupling_term:
          action[:7] = 0
          action_7 = np.zeros((7,))

        if self.params.force_term:
          f_a_3 = action[7:].reshape((self.params.a_dim,self.opti.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          observation_next, reward, done, suc = self.env.step(action_7, f_7, None, reset_flag)
        else:
          observation_next, reward, done, suc = self.env.step(action_7, None, None, reset_flag)

        reset_flag = False
        observation_c = np.copy(observation_next).reshape((-1,))
        while not done:
          coupling_ = None
          observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, coupling_, reset_flag)

        observation_next = np.reshape(observation_next,(-1,))
        print("Tesing","taskid",self.TaskId,"ep_",ep_," action",action[:7],"action_",action_[:7],"reward",reward,"suc",suc,"dmp dist",np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]), self.env.robotCurrentStatus()[:3],self.env.dmp.goal[:3])

        observation  = observation_next

        while len(self.env.real_traj_list) < self.env.dmp.timesteps:
            self.env.real_traj_list.append(self.env.real_traj_list[-1])
        print("self.env.real_tral_list",len(self.env.real_traj_list))
        GT_traj = np.array(self.env.real_traj_list)
        print("GT_traj",GT_traj)
        GT_traj = GT_traj.reshape((-1,))

        print("reward",reward)
        print("initial_pose",initial_pose)
        if suc:
          self.agent.store_transition_locally(initial_pose, GT_traj, observation_init, action_origin, reward, done, self.task_vec)

        if done:
          total_suc += float(suc)
          recordGif = self.params.recordGif
          if recordGif:
            classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
            recordGif_dir = os.path.join('./gif_dir_exp',str(self.params.task_id))
            if not os.path.exists(recordGif_dir):
              os.makedirs(recordGif_dir)
            imageio.mimsave(os.path.join(recordGif_dir,str(self.params.task_id)+'_'+str(ep_+0)+'.gif'),self.env.obs_list)
          break

    perf = total_suc / float(self.params.max_ep_test)
    print("success performance",perf)

    if self.params.writer is not None:
      self.params.writer.add_scalar("demon/suc",perf,eval_step)

  def imitate_single(self,task_id):
    save_top_dir = "/juno/group/linshao/ConceptLearning"
    save_sub_dir = os.path.join(save_top_dir,str(task_id))
    len_task_example = {5:2000,8:2000,11:2000,12:2000,13:1300,15:2000,16:1200,20:1200,27:900,40:2000,41:1300,42:2000,43:2000,44:1200,45:2000,46:1000,47:1650,53:2000,86:2000,87:2000,89:2000,93:2000,94:2000,96:1700,101:2000,104:2000,106:1800,118:1150,148:2000,171:2000}
    task_example_dict = [line for line in os.listdir(save_sub_dir) if int(line) < len_task_example[task_id]] 
    task_num_example = len(task_example_dict[task_id])
  
    iter_i = 0
    while True:
      task_example_list = []
      task_example_index_bs = np.random.randint(1,task_num_example,self.params.batch_size)
      for task_example_index in task_example_index_bs:
         file_path = os.path.join(save_top_dir,str(task_id),str(task_example_dict[task_example_index]),"example.npy")
         task_example = np.load(file_path)
         task_example_list.append(task_example)
      imitate_memory = np.array(task_example_list)
      self.agent.imitate_learn_v3(imitate_memory)      
      iter_i = iter_i + 1

      if iter_i % 1000 == 0:
        self.agent.save_model_master(iter_i)
        perf=self.test_master()
        print("start to evaluateion $$$$$$$$$$$$$$$$$$$$$")
        np.savetxt('up_per_'+str(task_id)+'.txt',np.array([perf]),fmt="%1.3f")
      if iter_i > 1001:
        break


 
  def imitate_multiple(self):
    save_top_dir = "/juno/group/linshao/ConceptLearning"
    task_list = [5,13,20,40,42,43,44,45,46,47,86,87,93,94,101,106,118]
    len_task_example = {45:2000,13:1300,12:1000,16:1200,46:1000,40:2000,42:2000,93:2000,94:2000,96:1700,20:1200,53:2000,27:900,86:2000,87:2000,104:2000,5:2000,171:2000,47:1650,101:2000,118:1150,44:1200,106:1800,27:1000,41:1300,43:2000,148:2000}
    task_example_dict = {}
    task_perf = [0.1,#5
                 0.1,#13
                 0.1,#20
                 0.1,#40
                 0.1,#42
                 0.1,#43
                 0.1,#44
                 0.1,#45
                 0.1,#46,
                 0.1,#47,
                 0.1,#86,
                 0.1,#87
                 0.1,#93
                 0.1,#94
                 0.1,#101
                 0.1,#106
                 0.1]#118
    task_perf_up = [0.56,#5
                 0.87,#13
                 0.68,#20
                 0.97,#40
                 0.79,#42
                 0.76,#43
                 0.99,#44
                 0.86,#45
                 0.84,#46 
                 1.0,#47,
                 0.74,#86,
                 1.0,#87
                 0.72,#93
                 1.0,#94
                 1.0,#101
                 0.38,#106
                 0.39]#118
    task_num_example_dict = {}
    task_example_dict = {}
    for b_task in task_list:
      save_sub_dir = os.path.join(save_top_dir,str(b_task))
      task_example_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) < len_task_example[b_task]]
      task_num_example = len(task_example_dict[str(b_task)])
      print("task_example_dict",task_example_dict[str(b_task)])
      print("task_num_eaxmpes",task_num_example)
      task_num_example_dict[str(b_task)] = task_num_example
    print("task_num_examples",task_num_example_dict)
    task_perf_upbound = np.array(task_perf_up)
    task_perf_cur = task_perf_upbound * 0.1
    task_id_list = np.array(task_list)

    iter_i = 0
    
    while True:
      pro = task_perf_cur / task_perf_upbound
      print("pro",pro)
      pro = np.minimum(pro,0.95 * np.ones_like(pro))
      pro = 1.05 - pro
      pro = softmax(pro)
      print(len(task_id_list),len(pro),pro)
      task_index_bs = np.random.choice(task_id_list,self.params.batch_size,p=pro)
      task_example_list = []
      print("task_index_bs",task_index_bs)
      for b_index in task_index_bs:
        task_index = b_index
        task_index_bs1 = np.random.randint(1,task_num_example_dict[str(b_index)])
        file_path = os.path.join(save_top_dir,str(task_index),str(task_example_dict[str(task_index)][task_index_bs1]),"example.npy")
        task_example = np.load(file_path)
        task_example_list.append(task_example)

      imitate_memory = np.array(task_example_list)
      self.agent.imitate_learn_v3(imitate_memory)
      iter_i = iter_i + 1
      if iter_i % 1000 == 0:
        self.agent.save_model_master(iter_i)
        #self.imitate_evaluation()
        print("start to evaluateion $$$$$$$$$$$$$$$$$$$$$")
        for idx, b_task in enumerate(task_list):
           self.TaskId = b_task
           print("evaluation task id",self.TaskId)
           perf = self.test_master()
           task_perf_cur[idx] = perf
        print("current perf", task_perf_cur)

  def imitate(self):
    save_top_dir = "/juno/group/linshao/ConceptLearning"
    task_list = [5,13,20,27,40,41,42,43,44,47,86,87,93,94,101,104,106,118,171]#[5,13,20,27,40,41,42,43,44,47,86,87,93,94,101,104,106,118,171] [45,45,43,16,12,46,5,13,27,40,41,42,43,44,47,93,94,93,94,86,86,87,101,104,106,118,171,96,148,20,27,53,43,45,44,41]
    len_task_example = {45:2000,13:1300,12:1000,16:1200,46:1000,40:2000,42:2000,93:2000,94:2000,96:1700,20:1200,53:2000,27:900,86:2000,87:2000,104:2000,5:2000,171:2000,47:1650,101:2000,118:1150,44:1200,106:1800,27:1000,41:1300,43:2000,148:2000}
    task_upbound_perm = {}
    print(task_list)
    task_num_example_list = []
    task_example_dict = {}
    for b_task in task_list:
      save_sub_dir = os.path.join(save_top_dir,str(b_task))
      print("save_sub_dir",save_sub_dir)
      task_example_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) < len_task_example[b_task]]
      task_num_example = len(task_example_dict[str(b_task)])
      print("task_example_dict",task_example_dict[str(b_task)])
      print("task_num_eaxmpes",task_num_example)
      task_num_example_list.append(task_num_example)
    print("task_num_examples",task_num_example_list)
    iter_i = 0
    while True:
     print("iter_i",iter_i)
     task_index_bs = np.random.randint(0,len(task_list),self.params.batch_size)
     #for b_task in range(len(task_list)):
     task_example_list = []
     print("task_index_bs",task_index_bs)
     for b_index in task_index_bs:
        task_index = task_list[b_index]
        task_index_bs1 = np.random.randint(1,task_num_example_list[b_index]) 
        file_path = os.path.join(save_top_dir,str(task_index),str(task_example_dict[str(task_index)][task_index_bs1]),"example.npy") 
        task_example = np.load(file_path)
        task_example_list.append(task_example)
        
     imitate_memory = np.array(task_example_list)
     self.agent.imitate_learn_v3(imitate_memory)
     iter_i = iter_i + 1
     if iter_i % 500 == 0:
       self.agent.save_model_master(iter_i)
       self.imitate_evaluation()
       print("start to evaluateion $$$$$$$$$$$$$$$$$$$$$")


  def imitate_evaluation(self):
    save_top_dir = "/juno/group/linshao/ConceptLearning"
    task_list = [5]#[45,45,43,16,12,46,5,13,27,40,41,42,43,44,47,93,94,93,94,86,86,87,101,104,106,118,171,96,148,20,27,53,43,45,44,41]
    len_task_example = {45:2000,13:1300,12:1000,16:1200,46:1000,40:2000,42:2000,93:2000,94:2000,96:1700,20:1200,53:2000,27:900,86:2000,87:2000,104:2000,5:2000,171:2000,47:1650,101:2000,118:1150,44:1200,106:1800,27:1000,41:1300,43:2000,148:2000}
    #print(task_list)
    task_num_example_list = []
    task_example_dict = {}
    for b_task in task_list:
      save_sub_dir = os.path.join(save_top_dir,str(b_task))
      print("save_sub_dir",save_sub_dir)
      task_example_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) > len_task_example[b_task]*0.9 and int(line) < len_task_example[b_task]]
      task_num_example = len(task_example_dict[str(b_task)])
      print("task_example_dict",task_example_dict[str(b_task)])
      print("task_num_eaxmpes",task_num_example)
      task_num_example_list.append(task_num_example)

    print("task_num_examples",task_num_example_list)

    iter_i = 0
    for iter_i in range(10):
    #while True:
     print("iter_i_evaluate",iter_i)
     task_index_bs = np.random.randint(0,len(task_list),self.params.batch_size)
     #for b_task in range(len(task_list)):
     if 1:
      task_example_list = []
      #task_index_bs = [b_task] * self.params.batch_size
      print("task_index_bs_evaluation",task_index_bs)
      for b_index in task_index_bs:
         task_index = task_list[b_index]
         task_index_bs1 = np.random.randint(1,task_num_example_list[b_index]) 
         file_path = os.path.join(save_top_dir,str(task_index),str(task_example_dict[str(task_index)][task_index_bs1]),"example.npy") 
         task_example = np.load(file_path)
         task_example_list.append(task_example)
      imitate_memory = np.array(task_example_list)
      self.agent.imitate_learn_v3_evaluation(imitate_memory)
      #iter_i = iter_i + 1

  def test_master(self,eval_step=0,restore_path=None):
    total_suc = 0
    self.p_id.__del__()
    del self.env
    self.env = Enver(self.opti,self.TaskId,self.params).env#self.env_list[self.TaskId]

    if restore_path is not None:
      self.agent.restore_master(eval_step,restore_path)
      print("testing master from restoring sth",restore_path)

    for ep_ in range(self.params.max_ep_test):
      observation = self.env.reset()
      observation = np.reshape(observation,(-1,))
      reset_flag = True

      self.task_vec = np.load("../Languages/"+str(self.TaskId)+'.npy')
      while True:
        action = np.zeros((50 * 7,))
        # Added exploration noise
        action_ = self.agent.choose_action_master(observation, self.task_vec)
        if self.params.force_term:
          action = action_
        else:
          action[:7] = action_[:7]
        
        action = action.clip(-0.5,0.5)
        action[3:6] = (action[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
        action[1] += 0.1
        action_7 = np.zeros((7,))
        action_7[:7] = action[:7]

        f_7 = np.zeros((self.opti.traj_timesteps,7))

        if self.params.force_term:
          f_a_3 = action[7:].reshape((self.params.a_dim,self.opti.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          observation_next, reward, done, suc = self.env.step(action_7, f_7, None, reset_flag)
        else:
          observation_next, reward, done, suc = self.env.step(action_7, None, None, reset_flag)

        reset_flag = False
        observation_c = np.copy(observation_next).reshape((-1,))
  
        while not done:
          coupling_ = None
          observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, coupling_, reset_flag)
        #print("Tesing","taskid",self.TaskId,"ep_",ep_," action",action[:7],"action_",action_[:7],"reward",reward,"suc",suc,"dmp dist",np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]), self.env.robotCurrentStatus()[:3],self.env.dmp.goal[:3])

        observation  = observation_next

        if done:
          total_suc += float(suc)
          recordGif = self.params.recordGif
          if recordGif:
            classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
            recordGif_dir = os.path.join('./gif_dir',str(self.params.task_id))
            if not os.path.exists(recordGif_dir):
              os.makedirs(recordGif_dir)
            imageio.mimsave(os.path.join(recordGif_dir,str(self.params.task_id)+'_'+str(ep_)+'.gif'),self.env.obs_list)
          break
    perf = total_suc / float(self.params.max_ep_test)
    print("success performance",perf)
    if self.params.writer is not None:
      self.params.writer.add_scalar("test_master/suc",perf,eval_step)
    return perf

if __name__ == '__main__':
  print("%%%%%%%%%%%%##########################")
  print("OPTI",opti.task_id,"force_term",opti.force_term,"coupling_term",opti.coupling_term,"only_coupling",opti.only_coupling_term)
  params = Params(TDI=opti.TDI,recordGif=opti.recordGif,\
                  action_penalty=opti.action_penalty,\
                  videoReward=opti.videoReward,\
                  only_coupling_term=opti.only_coupling_term,\
                  only_force_term=opti.only_force_term,\
                  use_cem=opti.use_cem,\
                  coupling_term=opti.coupling_term,\
                  start_learning_coupling=opti.start_learning_coupling,\
                  rotation_max_action = opti.rotation_max_action,\
                  traj_lr=opti.traj_lr,
                  traj_timesteps=opti.traj_timesteps,
                  batch_size=opti.batch_size,
                  test=opti.test,
                  exp_name=opti.exp_name,
                  max_ep_test=50,
                  force_term=opti.force_term,
                  task_id=opti.task_id,
                  gui=opti.gui,
                  start_learning=opti.start_learning,
                  explore_var=opti.explore_var)
  print("opti.test",opti.test,"TDI",opti.TDI)
  worker = Worker(wid=0,opti=opti,params=params)
  if not opti.test:
    worker.train(opti.restore_step, opti.restore_path)
  else:
    if opti.TDI == 0:
      worker.test(opti.restore_step, opti.restore_path)
    elif opti.TDI == 1:
      print("demonstrating")
      worker.demonstrate(opti.restore_step, opti.restore_path)
    elif opti.TDI == 2:
      worker.imitate_multiple()#imitate_single(int(opti.task_id))
    elif opti.TDI == 4:
      worker.imitate_single(int(opti.task_id))
    elif opti.TDI == 3:
      print("test master!")
      worker.test_master(opti.restore_step, opti.restore_path)
    else:
      print("sth wrong")
