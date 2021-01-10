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
DMP_DIR = os.path.join(BASE_DIR,'../deepDynamicMovementPrimitives')
sys.path.insert(0,DMP_DIR)
from ddmp import DDMP  as DMP

sys.path.insert(0, "../../simulation")

import bullet_client as bc

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

#####
from actor import Actor
from critic import Critic
#from actor_coupling import Actor_C
#from critic_coupling import Critic_C
from master import Master

transforms = transforms.Compose([
  transforms.ToPILImage(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  ])


class Agent(object):
  def __init__(self, params):
    self.device = torch.device('cuda')
    self.policy_freq = 2

    self.params = params
    print("self.params.force_term",self.params.force_term)
    self.memory = np.zeros((self.params.mem_capacity, self.params.a_dim + self.params.traj_timesteps * self.params.a_dim + self.params.state_dim + self.params.a_dim + self.params.traj_timesteps * self.params.a_dim + 2 + self.params.task_dim), dtype=np.float32)
    print("self.memory",self.memory.shape,self.params.a_dim,self.params.traj_timesteps * self.params.a_dim,self.params.state_dim ,self.params.traj_timesteps * self.params.a_dim,self.params.task_dim)
    self.pointer = 0
    self.step = 0

    self.step_traj = 0
    self.imitate_step = 0
    self.imitate_step_eval = 0

    self.actor = Actor(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.params.a_lr)
    
    self.critic = Critic(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.params.a_lr)

    self.master = Master(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.master_optimizer = torch.optim.Adam(self.master.parameters(), self.params.traj_lr)


  def store_transition(self, init_pose, gt_traj, s, a, r, t, task_vec):
    transition = np.hstack((init_pose, gt_traj, s, a, [r, t], task_vec))
    index = self.pointer % self.params.mem_capacity
    self.memory[index, :] = transition
    self.pointer += 1

  def store_transition_locally(self, init_pose, gt_traj, s, a, r, t, task_vec):
    transition = np.hstack((init_pose, gt_traj, s, a, [r, t], task_vec))
    #print("gt_traj",gt_traj)
    self.pointer += 1
    save_top_dir = "/juno/group/linshao/ConceptLearning"
    save_sub_top_dir = os.path.join(save_top_dir,str(self.params.task_id))
    if not os.path.exists(save_sub_top_dir):
      os.makedirs(save_sub_top_dir)
    save_file_top_dir = os.path.join(save_sub_top_dir,str(self.pointer))
    if not os.path.exists(save_file_top_dir):
      os.makedirs(save_file_top_dir)
    save_file_path = os.path.join(save_file_top_dir,'example.npy')
    np.save(save_file_path, transition)
    print(save_file_path)

  def choose_action(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    init_action = self.actor(state, task_vec).cpu().data.numpy().flatten()
    if self.params.use_cem:    
      init_action = self.cem_choose_action(init_action, state, task_vec)
    return init_action

  def choose_action_master(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    init_action = self.master(state, task_vec).cpu().data.numpy().flatten()
    #if self.params.use_cem:    
    #  init_action = self.cem_choose_action(init_action, state, task_vec)
    return init_action

  def learn(self):
    self.step += 1

    if self.pointer > self.params.mem_capacity:
      indices = np.random.choice(self.params.mem_capacity,size=self.params.batch_size)
    else:
      indices = np.random.choice(self.pointer,size=self.params.batch_size)

    bt = self.memory[indices, :]

    index1 = self.params.a_dim
    b_traj_start = bt[:, : index1]

    index2 = index1 + self.params.traj_timesteps * self.params.a_dim
    b_gt_traj = bt[:, index1 : index2]
    
    index3 = index2 + self.params.state_dim
    bs = bt[:, index2: index3]
     
    index4 = index3 + self.params.a_dim * self.params.traj_timesteps + self.params.a_dim 
    ba = bt[:, index3: index4]

    index5 = index4 + 1
    br = bt[:, index4: index5]

    index6 = index5 + 1
    bd = bt[:, index5: index6]

    index7 = index6 + self.params.task_dim
    btask = bt[:, index6: index7]

    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    action = ba.copy().reshape((self.params.batch_size,-1))
    if not self.params.force_term:
      action = action[:,:self.params.a_dim]
    if self.params.only_force_term:
      action = action[:,self.params.a_dim:]

    action = torch.FloatTensor(action).to(self.device)
    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)
    reward = br.copy().reshape((self.params.batch_size,-1))
    reward = torch.FloatTensor(reward).to(self.device)

    target_q = reward
    current_q = self.critic(state, task_vec, action)
    critic_loss = F.mse_loss(current_q, target_q)

    self.params.writer.add_scalar('train_critic/critic_loss',critic_loss, self.step)
    self.params.writer.add_scalar('train_critic/reward',reward.mean(), self.step)
    self.params.writer.add_scalar('train_critic/current_Q',current_q.mean(), self.step)
    self.params.writer.add_scalar('train_critic/reward_max',reward.max(), self.step)
    self.params.writer.add_scalar('train_critic/current_Q_max',current_q.max(), self.step)
     
    # optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    if self.step % self.policy_freq == 0:
      print("updating actor")
      actor_loss = -self.critic(state, task_vec, self.actor(state, task_vec)).mean()
      self.params.writer.add_scalar('train_actor/actor_Q',-actor_loss, self.step)

      # Optimize the actor 
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()


  def cem_choose_action(self,init_action,s, task):
    # Initialize mean and stanford deviation
    theta_mean = init_action#np.zeros(A_DIM)
    theta_std = np.ones_like(init_action) * 0.1
    batch_size = 100 # number of samples per batch
    elite_frac = 0.1 # fraction of samples used as elite set
    n_elite = int(batch_size * elite_frac)
    extra_std = np.ones_like(init_action) * 0.01
    extra_std[2] = 0.01
    extra_decay_time = 2
    n_iter = 3
    for itr in range(n_iter):
      extra_cov = max(1.0 - itr / extra_decay_time, 0) * np.array(extra_std **2)
      thetas = np.random.multivariate_normal(mean=theta_mean,
                                         cov=np.diag(np.array(theta_std**2) + extra_cov),
                                         size=batch_size).clip(-self.params.max_action, self.params.max_action)
      rewards = self.cem_evaluate(thetas,s, task)
      rewards = np.squeeze(rewards)
      elite_inds = rewards.argsort()[-n_elite:]
      elite_thetas = thetas[elite_inds]

      # Get elite parameters
      theta_mean = elite_thetas.mean(axis=0)
      theta_std = elite_thetas.std(axis=0)
    self.mean = theta_mean
    return (theta_mean).clip(-self.params.max_action, self.params.max_action)

  def cem_evaluate(self, mean, state, task_vec):
    pop_size = len(mean)
    actions = torch.FloatTensor(mean).to(self.device)
    states = state.repeat(pop_size, 1, 1, 1) 
    task_vecs = task_vec.repeat(pop_size, 1)
    print("states",states.size(),"actions",actions.size(),"task_vecs",task_vecs.size())
    Qs = self.critic(states, task_vecs, actions)
    return Qs.cpu().data.numpy().flatten()

  def restore(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_critic = os.path.join(restore_path, 'critic_'+str(step)+'_model.pth.tar')
    self.critic.load_state_dict(torch.load(save_path_critic))

    save_path_actor = os.path.join(restore_path, 'actor_'+str(step)+'_model.pth.tar')
    self.actor.load_state_dict(torch.load(save_path_actor))


  def save_model(self, step):
    save_path_critic = os.path.join(self.params.model_dir, 'critic_'+str(step)+'_model.pth.tar')
    torch.save(self.critic.state_dict(), save_path_critic)

    save_path_actor = os.path.join(self.params.model_dir, 'actor_'+str(step)+'_model.pth.tar')
    torch.save(self.actor.state_dict(), save_path_actor)

    save_path_master = os.path.join(self.params.model_dir, 'master_'+str(step)+'_model.pth.tar')
    torch.save(self.master.state_dict(), save_path_master)

  def save_model_master(self, step):
    save_path_master = os.path.join(self.params.model_dir, 'master_'+str(step)+'_model.pth.tar')
    torch.save(self.master.state_dict(), save_path_master)

  def restore_master(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_master = os.path.join(restore_path, 'master_'+str(step)+'_model.pth.tar')
    self.master.load_state_dict(torch.load(save_path_master))


  def imitate_learn(self,imitate_memory):
    bt = imitate_memory

    index1 = self.params.a_dim
    b_traj_start = bt[:, : index1]

    index2 = index1 + self.params.traj_timesteps * self.params.a_dim
    b_gt_traj = bt[:, index1 : index2]

    index3 = index2 + self.params.state_dim
    bs = bt[:, index2: index3]

    index4 = index3 + self.params.a_dim * self.params.traj_timesteps + self.params.a_dim
    ba = bt[:, index3: index4]

    index5 = index4 + 1
    #br = bt[:, index4: index5]

    index6 = index5 + 1
    #bd = bt[:, index5: index6]

    index7 = index6 + self.params.task_dim
    btask = bt[:, index6: index7]


    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)
    
    for ik in range(1):
      self.imitate_step += 1
      offset = 0
      pred_action = self.master(state, task_vec)
       
      gt_action_label = torch.FloatTensor(ba).to(self.device) 
      imitate_loss = F.mse_loss(gt_action_label[:,:7], pred_action[:,:7]) + 0.01 * F.mse_loss(gt_action_label[:,7:], pred_action[:,7:])#imitate_goal_loss + imitate_force_loss
      self.master_optimizer.zero_grad()
      imitate_loss.backward()
      self.master_optimizer.step()

      gt_force = gt_action_label[:,7:].reshape((-1,7,49))
      pred_force = pred_action[:,7:].reshape((-1,7,49))
      self.params.writer.add_scalar('train_imitate/imitate_loss',imitate_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_0',pred_action[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_1',pred_action[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_2',pred_action[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_3',pred_action[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_4',pred_action[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_5',pred_action[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_6',pred_action[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_force',pred_action[:,7:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_0',gt_action_label[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_1',gt_action_label[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_2',gt_action_label[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_3',gt_action_label[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_4',gt_action_label[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_5',gt_action_label[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_6',gt_action_label[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_0',gt_force[:,0,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_1',gt_force[:,1,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_2',gt_force[:,2,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_3',gt_force[:,3,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_4',gt_force[:,4,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_5',gt_force[:,5,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_0',pred_force[:,0,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_1',pred_force[:,1,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_2',pred_force[:,2,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_3',pred_force[:,3,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_4',pred_force[:,4,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_5',pred_force[:,5,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_0',(gt_action_label[:,0]-pred_action[:,0]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_1',(gt_action_label[:,1]-pred_action[:,1]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_2',(gt_action_label[:,2]-pred_action[:,2]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_3',(gt_action_label[:,3]-pred_action[:,3]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_4',(gt_action_label[:,4]-pred_action[:,4]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_5',(gt_action_label[:,5]-pred_action[:,5]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_6',(gt_action_label[:,6]-pred_action[:,6]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_force',(gt_action_label[:,7:]-pred_action[:,7:]).reshape((-1,49,7))[:,:,0].mean(),self.imitate_step)
 

  def imitate_learn_v2(self,imitate_memory):
    bt = imitate_memory

    index1 = self.params.a_dim
    b_traj_start = bt[:, : index1]

    index2 = index1 + self.params.traj_timesteps * self.params.a_dim
    b_gt_traj = bt[:, index1 : index2]

    index3 = index2 + self.params.state_dim
    bs = bt[:, index2: index3]

    index4 = index3 + self.params.a_dim * self.params.traj_timesteps + self.params.a_dim
    ba = bt[:, index3: index4]

    index5 = index4 + 1
    #br = bt[:, index4: index5]

    index6 = index5 + 1
    #bd = bt[:, index5: index6]

    index7 = index6 + self.params.task_dim
    btask = bt[:, index6: index7]


    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    gt_action_label = torch.FloatTensor(ba).to(self.device) 
    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)
    
    real_traj = b_gt_traj.reshape((self.params.batch_size,self.params.traj_timesteps,self.params.a_dim)) 
    #print(real_traj.shape)
    real_traj_goal = torch.FloatTensor(real_traj[:,-1] - real_traj[:,0]).to(self.device) 

    for ik in range(1):
      self.imitate_step += 1
      offset = 0
      pred_action = self.master(state, task_vec)
      if 1:
        master_action = self.master(state, task_vec).cpu().detach().numpy()
        gt_goal_list = []
        gt_force_list = []
        gt_dmp_goal_list = []

        for bi in range(self.params.batch_size):
          f_7 = np.zeros((self.params.traj_timesteps,7))
          f_a_3 = ba[bi,7:].reshape((self.params.a_dim,self.params.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          action_7 = np.zeros((7,))
          action_7 = ba[bi,:7].clip(-0.5,0.5)
          action_7[3:6] = (action_7[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
          goal_gt_action = action_7 + b_traj_start[0]
          dmp_gt_action = DMP(None,n_dmps=self.params.a_dim,goal=goal_gt_action,start=b_traj_start[0],force=f_7,timesteps=self.params.traj_timesteps)
          traj_gt_action = dmp_gt_action.rollout()[0] # (49,7)
          #print("start",np.array((traj_gt_action[-1]-traj_gt_action[0])),"goal",action_7) 
          dmp_goal_gt_action = np.array((traj_gt_action[-1]-traj_gt_action[0]))
          gt_dmp_goal_list.append(dmp_goal_gt_action)

        gt_dmp_goal_array = np.array(gt_dmp_goal_list)
        gt_goal_array = np.array(gt_goal_list)
        gt_force_array = np.array(gt_force_list)
        gt_goal_force = np.hstack([gt_goal_array, gt_force_array])
        gt_dmp_goal_label = torch.FloatTensor(gt_dmp_goal_array).to(self.device)
        gt_goal_label = torch.FloatTensor(gt_goal_array).to(self.device)
        gt_force_label = torch.FloatTensor(gt_force_array).to(self.device)
      imitate_loss = F.mse_loss(real_traj_goal, pred_action[:,:7]) + F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]) + 0.01 * F.mse_loss(torch.zeros_like(gt_action_label[:,7:]), pred_action[:,7:])#imitate_goal_loss + imitate_force_loss
 
      self.master_optimizer.zero_grad()
      imitate_loss.backward()
      self.master_optimizer.step()

      #print("imitate_loss", imitate_loss, " imitate_step", self.imitate_step)
      #print((gt_action_label[:,7:]-pred_action[:,7:]).size())
      gt_force = gt_action_label[:,7:].reshape((-1,7,49))
      pred_force = pred_action[:,7:].reshape((-1,7,49))
      self.params.writer.add_scalar('train_imitate/imitate_loss',imitate_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_0',pred_action[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_1',pred_action[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_2',pred_action[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_3',pred_action[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_4',pred_action[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_5',pred_action[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_6',pred_action[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_action_force',pred_action[:,7:].mean(),self.imitate_step)

      self.params.writer.add_scalar('train_imitate/real_goal_0',real_traj_goal[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_1',real_traj_goal[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_2',real_traj_goal[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_3',real_traj_goal[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_4',real_traj_goal[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_5',real_traj_goal[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/real_goal_6',real_traj_goal[:,6].mean(),self.imitate_step)
 
      self.params.writer.add_scalar('train_imitate/gt_action_0',gt_action_label[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_1',gt_action_label[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_2',gt_action_label[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_3',gt_action_label[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_4',gt_action_label[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_5',gt_action_label[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_action_6',gt_action_label[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_0',gt_force[:,0,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_1',gt_force[:,1,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_2',gt_force[:,2,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_3',gt_force[:,3,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_4',gt_force[:,4,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/gt_force_5',gt_force[:,5,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_0',pred_force[:,0,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_1',pred_force[:,1,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_2',pred_force[:,2,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_3',pred_force[:,3,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_4',pred_force[:,4,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_force_5',pred_force[:,5,:].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_0',(gt_action_label[:,0]-pred_action[:,0]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_1',(gt_action_label[:,1]-pred_action[:,1]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_2',(gt_action_label[:,2]-pred_action[:,2]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_3',(gt_action_label[:,3]-pred_action[:,3]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_4',(gt_action_label[:,4]-pred_action[:,4]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_5',(gt_action_label[:,5]-pred_action[:,5]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_6',(gt_action_label[:,6]-pred_action[:,6]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate/pred_error_action_force',(gt_action_label[:,7:]-pred_action[:,7:]).reshape((-1,49,7))[:,:,0].mean(),self.imitate_step)


 
  def imitate_learn_v3(self,imitate_memory):
    bt = imitate_memory

    index1 = self.params.a_dim
    b_traj_start = bt[:, : index1]

    index2 = index1 + self.params.traj_timesteps * self.params.a_dim
    b_gt_traj = bt[:, index1 : index2]

    index3 = index2 + self.params.state_dim
    bs = bt[:, index2: index3]

    index4 = index3 + self.params.a_dim * self.params.traj_timesteps + self.params.a_dim
    ba = bt[:, index3: index4]

    index5 = index4 + 1
    #br = bt[:, index4: index5]

    index6 = index5 + 1

    index7 = index6 + self.params.task_dim
    btask = bt[:, index6: index7]
    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    gt_action_label = torch.FloatTensor(ba).to(self.device) 
    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)

    #print("self.params.batch_size",self.params.batch_size)
    real_traj = b_gt_traj.reshape((self.params.batch_size,self.params.traj_timesteps,self.params.a_dim))
    #print(real_traj.shape)
    real_traj_goal_array = real_traj[:,-1] - real_traj[:,0]
    if self.params.rotation_max_action < 1e-6:
      real_traj_goal_array[:,3:6] = 0.0   
    real_traj_goal = torch.FloatTensor(real_traj_goal_array).to(self.device)

    for ik in range(1):
      self.imitate_step += 1
      offset = 0
      pred_action = self.master(state, task_vec)
      pred_force = pred_action[:,7:].reshape((self.params.batch_size,self.params.a_dim,self.params.traj_timesteps)).transpose(1,2)*50.0
      weights_force = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
      weights_force = np.expand_dims(weights_force,axis=0)
      pred_force = pred_force * torch.FloatTensor(weights_force).to(self.device) ### (bn, 49, 7)
      pred_force[:,:,3:6].clamp(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)

      if 1:
        master_action = self.master(state, task_vec).cpu().detach().numpy()
        gt_goal_list = []
        gt_force_list = []
        gt_dmp_goal_list = []
        traj_error = 0
        for bi in range(self.params.batch_size):
          f_7 = np.zeros((self.params.traj_timesteps,7))
          f_a_3 = ba[bi,7:].reshape((self.params.a_dim,self.params.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          action_7 = np.zeros((7,))
          action_7 = ba[bi,:7].clip(-0.5,0.5)
          action_7[3:6] = (action_7[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
          goal_gt_action = action_7 + b_traj_start[0]
          dmp_gt_action = DMP(None,n_dmps=self.params.a_dim,goal=goal_gt_action,start=b_traj_start[0],force=f_7,timesteps=self.params.traj_timesteps)
          traj_gt_action = dmp_gt_action.rollout()[0] # (49,7)
          dmp_goal_gt_action = np.array((traj_gt_action[-1]-traj_gt_action[0]))
          if self.params.rotation_max_action < 1e-6:
            dmp_goal_gt_action[3:6] = 0.0
          gt_dmp_goal_list.append(dmp_goal_gt_action)
          
          dmp_force_label = DMP(None,n_dmps=self.params.a_dim,goal=real_traj[bi][-1],start=real_traj[bi][0],force=None,timesteps=self.params.traj_timesteps)
          force_label = dmp_force_label.imitate_path(real_traj[bi]) #(shape, (49,7))
          force_label[:,3:6].clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          gt_force_list.append(force_label)

        gt_dmp_goal_array = np.array(gt_dmp_goal_list)
        gt_goal_array = np.array(gt_goal_list)
        gt_dmp_goal_array[:,2] = 0.5 * gt_dmp_goal_array[:,2] + 0.5 * real_traj_goal_array[:,2]
        gt_force_array = np.array(gt_force_list)

        gt_dmp_goal_label = torch.FloatTensor(gt_dmp_goal_array).to(self.device)
        gt_goal_label = torch.FloatTensor(gt_goal_array).to(self.device)
        gt_force_label = torch.FloatTensor(gt_force_array).to(self.device) #gt_force_label torch.Size([20, 49, 7])
 
      vio_loss_count = ((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label) > torch.zeros_like(pred_action[:,:7])).sum()
      vio_loss = torch.max( (pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label),torch.zeros_like(pred_action[:,:7])).sum() 
      imitate_loss = F.mse_loss(real_traj_goal, pred_action[:,:7]) * 0.1 + F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]) + 0.01 * F.mse_loss(torch.zeros_like(gt_force_label), pred_force) #+ torch.max((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label), torch.zeros_like(pred_action[:,:7])).sum() * 100.0
      #imitate_loss = F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]) + 0.01 * F.mse_loss(torch.zeros_like(gt_force_label), pred_force) #+ torch.max((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label), torch.zeros_like(pred_action[:,:7])).sum() * 100.0
 
      self.master_optimizer.zero_grad()
      imitate_loss.backward()
      self.master_optimizer.step()

      pred_force_in_action = pred_action[:,7:].reshape((self.params.batch_size,self.params.a_dim,self.params.traj_timesteps)).transpose(1,2) # (bn, 49, 7)
      self.params.writer.add_scalar('train_imitate_loss/vio_loss_count',vio_loss_count, self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_loss',imitate_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_loss1',F.mse_loss(real_traj_goal, pred_action[:,:7]), self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_loss2',F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]), self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_loss3',0.001 * F.mse_loss(gt_force_label, pred_force), self.imitate_step)

      self.params.writer.add_scalar('train_pred_raw_action/pred_action_0',pred_action[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_1',pred_action[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_2',pred_action[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_3',pred_action[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_4',pred_action[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_5',pred_action[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_6',pred_action[:,6].mean(),self.imitate_step)

      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_0',(pred_force_in_action[:,:,0]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_1',(pred_force_in_action[:,:,1]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_2',(pred_force_in_action[:,:,2]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_3',(pred_force_in_action[:,:,3]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_4',(pred_force_in_action[:,:,4]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_5',(pred_force_in_action[:,:,5]).mean(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_force_in_action_6',(pred_force_in_action[:,:,6]).mean(),self.imitate_step)

      self.params.writer.add_scalar('train_pred_raw_action/pred_action_0_std',pred_action[:,0].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_1_std',pred_action[:,1].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_2_std',pred_action[:,2].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_3_std',pred_action[:,3].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_4_std',pred_action[:,4].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_5_std',pred_action[:,5].std(),self.imitate_step)
      self.params.writer.add_scalar('train_pred_raw_action/pred_action_6_std',pred_action[:,6].std(),self.imitate_step)
 

      self.params.writer.add_scalar('train_goal/gt_dmp_action_0',gt_dmp_goal_array[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_1',gt_dmp_goal_array[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_2',gt_dmp_goal_array[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_3',gt_dmp_goal_array[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_4',gt_dmp_goal_array[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_5',gt_dmp_goal_array[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_dmp_action_6',gt_dmp_goal_array[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_0',real_traj_goal[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_1',real_traj_goal[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_2',real_traj_goal[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_3',real_traj_goal[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_4',real_traj_goal[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_5',real_traj_goal[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_goal/gt_real_action_6',real_traj_goal[:,6].mean(),self.imitate_step)


      self.params.writer.add_scalar('train_force/real_force_0',gt_force_array[:,:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_1',gt_force_array[:,:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_2',gt_force_array[:,:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_3',gt_force_array[:,:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_4',gt_force_array[:,:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_5',gt_force_array[:,:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/real_force_6',gt_force_array[:,:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/pred_force_0',pred_force[:,:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/pred_force_1',pred_force[:,:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/pred_force_4',pred_force[:,:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/pred_force_5',pred_force[:,:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_force/pred_force_6',pred_force[:,:,6].mean(),self.imitate_step)


 
  def imitate_learn_v3_evaluation(self,imitate_memory):
    bt = imitate_memory

    index1 = self.params.a_dim
    b_traj_start = bt[:, : index1]

    index2 = index1 + self.params.traj_timesteps * self.params.a_dim
    b_gt_traj = bt[:, index1 : index2]

    index3 = index2 + self.params.state_dim
    bs = bt[:, index2: index3]

    index4 = index3 + self.params.a_dim * self.params.traj_timesteps + self.params.a_dim
    ba = bt[:, index3: index4]

    index5 = index4 + 1

    index6 = index5 + 1

    index7 = index6 + self.params.task_dim
    btask = bt[:, index6: index7]
    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    gt_action_label = torch.FloatTensor(ba).to(self.device) 
    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)

    #print("self.params.batch_size",self.params.batch_size)
    real_traj = b_gt_traj.reshape((self.params.batch_size,self.params.traj_timesteps,self.params.a_dim))
    #print(real_traj.shape)
    real_traj_goal_array = real_traj[:,-1] - real_traj[:,0]
    if self.params.rotation_max_action < 1e-6:
      real_traj_goal_array[:,3:6] = 0.0   
    real_traj_goal = torch.FloatTensor(real_traj_goal_array).to(self.device)

    self.imitate_step_eval = self.imitate_step

    for ik in range(1):
      self.imitate_step_eval += 1
      offset = 0
      pred_action = self.master(state, task_vec)
      pred_force = pred_action[:,7:].reshape((self.params.batch_size,self.params.a_dim,self.params.traj_timesteps)).transpose(1,2)*50.0
      weights_force = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
      weights_force = np.expand_dims(weights_force,axis=0)
      pred_force = pred_force * torch.FloatTensor(weights_force).to(self.device) ### (bn, 49, 7)
      pred_force[:,:,3:6].clamp(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)

      if 1:
        master_action = self.master(state, task_vec).cpu().detach().numpy()
        gt_goal_list = []
        gt_force_list = []
        gt_dmp_goal_list = []
        traj_error = 0
        for bi in range(self.params.batch_size):
          f_7 = np.zeros((self.params.traj_timesteps,7))
          f_a_3 = ba[bi,7:].reshape((self.params.a_dim,self.params.traj_timesteps)).transpose() * 50.0
          weights = np.linspace(1,0,self.params.traj_timesteps).reshape((self.params.traj_timesteps,1))
          f_7[:,:7] = f_a_3 * weights
          f_7[:,3:6] = (f_7[:,3:6]).clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          action_7 = np.zeros((7,))
          action_7 = ba[bi,:7].clip(-0.5,0.5)
          action_7[3:6] = (action_7[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
          goal_gt_action = action_7 + b_traj_start[0]
          dmp_gt_action = DMP(None,n_dmps=self.params.a_dim,goal=goal_gt_action,start=b_traj_start[0],force=f_7,timesteps=self.params.traj_timesteps)
          traj_gt_action = dmp_gt_action.rollout()[0] # (49,7)
          dmp_goal_gt_action = np.array((traj_gt_action[-1]-traj_gt_action[0]))
          if self.params.rotation_max_action < 1e-6:
            dmp_goal_gt_action[3:6] = 0.0
          gt_dmp_goal_list.append(dmp_goal_gt_action)
          
          dmp_force_label = DMP(None,n_dmps=self.params.a_dim,goal=real_traj[bi][-1],start=real_traj[bi][0],force=None,timesteps=self.params.traj_timesteps)
          force_label = dmp_force_label.imitate_path(real_traj[bi]) #(shape, (49,7))
          force_label[:,3:6].clip(-self.params.rotation_max_action/np.pi*50.0, self.params.rotation_max_action/np.pi*50.0)
          gt_force_list.append(force_label)

        gt_dmp_goal_array = np.array(gt_dmp_goal_list)
        gt_dmp_goal_array[:,2] = 0.5 * gt_dmp_goal_array[:,2] + 0.5 * real_traj_goal_array[:,2]
        gt_goal_array = np.array(gt_goal_list)
        gt_force_array = np.array(gt_force_list)

        gt_dmp_goal_label = torch.FloatTensor(gt_dmp_goal_array).to(self.device)
        gt_goal_label = torch.FloatTensor(gt_goal_array).to(self.device)
        gt_force_label = torch.FloatTensor(gt_force_array).to(self.device) #gt_force_label torch.Size([20, 49, 7])
 
      vio_loss_count = ((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label) > torch.zeros_like(pred_action[:,:7])).sum()
      vio_loss = torch.max( (pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label),torch.zeros_like(pred_action[:,:7])).sum() 
      imitate_loss = F.mse_loss(real_traj_goal, pred_action[:,:7]) * 0.1 + F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]) + 0.01 * F.mse_loss(torch.zeros_like(gt_force_label), pred_force) #+ torch.max((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label), torch.zeros_like(pred_action[:,:7])).sum() * 100.0
      
      #imitate_loss = F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]) + 0.01 * F.mse_loss(torch.zeros_like(gt_force_label), pred_force) #+ torch.max((pred_action[:,:7] - real_traj_goal) * (pred_action[:,:7] - gt_dmp_goal_label), torch.zeros_like(pred_action[:,:7])).sum() * 100.0
 
      pred_force_in_action = pred_action[:,7:].reshape((self.params.batch_size,self.params.a_dim,self.params.traj_timesteps)).transpose(1,2) # (bn, 49, 7)
      self.params.writer.add_scalar('eval_imitate_loss/vio_loss_count',vio_loss_count, self.imitate_step_eval)
      self.params.writer.add_scalar('eval_imitate_loss/imitate_loss',imitate_loss, self.imitate_step_eval)
      self.params.writer.add_scalar('eval_imitate_loss/imitate_loss1',F.mse_loss(real_traj_goal, pred_action[:,:7]), self.imitate_step_eval)
      self.params.writer.add_scalar('eval_imitate_loss/imitate_loss2',F.mse_loss(gt_dmp_goal_label, pred_action[:,:7]), self.imitate_step_eval)
      self.params.writer.add_scalar('eval_imitate_loss/imitate_loss3',0.001 * F.mse_loss(gt_force_label, pred_force), self.imitate_step_eval)

      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_0',pred_action[:,0].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_1',pred_action[:,1].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_2',pred_action[:,2].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_3',pred_action[:,3].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_4',pred_action[:,4].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_5',pred_action[:,5].mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_6',pred_action[:,6].mean(),self.imitate_step_eval)

      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_0',(pred_force_in_action[:,:,0]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_1',(pred_force_in_action[:,:,1]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_2',(pred_force_in_action[:,:,2]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_3',(pred_force_in_action[:,:,3]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_4',(pred_force_in_action[:,:,4]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_5',(pred_force_in_action[:,:,5]).mean(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_force_in_action_6',(pred_force_in_action[:,:,6]).mean(),self.imitate_step_eval)

      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_0_std',pred_action[:,0].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_1_std',pred_action[:,1].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_2_std',pred_action[:,2].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_3_std',pred_action[:,3].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_4_std',pred_action[:,4].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_5_std',pred_action[:,5].std(),self.imitate_step_eval)
      self.params.writer.add_scalar('eval_pred_raw_action/pred_action_6_std',pred_action[:,6].std(),self.imitate_step_eval)
