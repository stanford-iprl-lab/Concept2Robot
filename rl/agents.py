import copy
import numpy as np
import os
import sys
import time
np.set_printoptions(precision=4,suppress=False)
import cv2
import importlib
import glob
import imageio
import math
from tensorboardX import SummaryWriter
import datetime
import copy
import matplotlib.pyplot as plt

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
    self.memory = np.zeros((self.params.mem_capacity, self.params.a_dim + self.params.traj_timesteps * self.params.a_dim + self.params.state_dim + self.params.a_dim + self.params.traj_timesteps * self.params.a_dim + 2 + self.params.task_dim + self.params.a_dim), dtype=np.float32)
    print("self.memory",self.memory.shape,self.params.a_dim,self.params.traj_timesteps * self.params.a_dim,self.params.state_dim ,self.params.traj_timesteps * self.params.a_dim,self.params.task_dim)
    self.pointer = 0
    self.step = 0

    self.step_traj = 0
    self.imitate_step = 0
    self.imitate_step_eval = 0

    self.actor = Actor(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.params.a_lr)
    self.actor_goal_only = Actor(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)

    self.critic = Critic(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.params.a_lr)

    self.master = Master(self.params.state_dim, self.params.a_dim, self.params.task_dim, self.params.max_action, self.params).to(self.device)
    self.master_optimizer = torch.optim.Adam(self.master.parameters(), self.params.m_lr)

    self.pointer_feedback = 0
    self.step_feedback = 0
    self.tau = 0.01

  def store_transition(self, init_pose, gt_traj, s, a, r, t, task_vec, a_gt):
    transition = np.hstack((init_pose, gt_traj, s, a, [r, t], task_vec,a_gt))
    index = self.pointer % self.params.mem_capacity
    self.memory[index, :] = transition
    self.pointer += 1

  def store_transition_feedback(self, flag, s, a, r, t, task_vec):
    transition = np.hstack(([flag], s, a, [r, t], task_vec))
    index = self.pointer_feedback % self.params.mem_feedback_capacity
    self.memory_feedback[index, :] = transition
    self.pointer_feedback += 1

  def store_transition_locally(self, init_pose, gt_traj, s, a, r, t, task_vec, save_top_dir):
    transition = np.hstack((init_pose, gt_traj, s, a, [r, t], task_vec))
    save_sub_top_dir = save_top_dir
    if not os.path.exists(save_sub_top_dir):
      os.makedirs(save_sub_top_dir)
    save_file_top_dir = os.path.join(save_sub_top_dir,str(self.pointer))
    if not os.path.exists(save_file_top_dir):
      os.makedirs(save_file_top_dir)
    save_file_path = os.path.join(save_file_top_dir,'example.npy')
    np.save(save_file_path, transition)
    print(save_file_path)
    self.pointer += 1

  def store_transition_feedback_locally(self, timestep, flag, s, a, r, t, task_vec, save_top_dir):
    transition = np.hstack((flag, s, a, [r, t], task_vec))
    save_sub_top_dir = save_top_dir
    if not os.path.exists(save_sub_top_dir):
      os.makedirs(save_sub_top_dir)
    save_file_top_dir = os.path.join(save_sub_top_dir,str(self.pointer))
    if not os.path.exists(save_file_top_dir):
      os.makedirs(save_file_top_dir)
    save_file_path = os.path.join(save_file_top_dir,'example_'+str(timestep)+'_feedback.npy')
    np.save(save_file_path, transition)
    print(save_file_path)

  def choose_action(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    goal_action, force_action = self.actor(state, task_vec)
    goal_action = goal_action.cpu().data.numpy().flatten()
    force_action = force_action.cpu().data.numpy()[0]
    return goal_action, force_action

  def choose_action_goal_only(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    goal_action, _ = self.actor(state, task_vec)
    return goal_action.cpu().data.numpy().flatten()

  def choose_action_master(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    goal_action, force_action = self.master(state, task_vec)
    goal_action = goal_action.cpu().data.numpy().flatten()
    force_action = force_action.cpu().data.numpy()[0]
    return goal_action, force_action

  def choose_action_feedback(self, state, task_vec):
    state = state.reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    task_vec = torch.FloatTensor(task_vec.reshape((-1,self.params.task_dim))).to(self.device)
    assert state.size()[0] == 4 * task_vec.size()[0]
    self.actor_feedback.eval()
    init_action = self.actor_feedback(state, task_vec).cpu().data.numpy().flatten()
    return init_action

  def learn(self):
    self.step += 1
    if self.pointer > self.params.mem_capacity:
      indices = np.random.choice(self.params.mem_capacity, size=self.params.batch_size)
    else:
      indices = np.random.choice(self.pointer, size=self.params.batch_size)

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
    
    index8 = index7 + self.params.a_dim
    ba_gt = bt[:, index7: index8]

    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)
    action = ba.copy().reshape((self.params.batch_size,-1))
    action_gt = ba_gt.copy().reshape((self.params.batch_size,-1))
    
    if not self.params.force_term:
      action = action[:,:self.params.a_dim]

    action = torch.FloatTensor(action).to(self.device)
    action_gt = torch.FloatTensor(action_gt).to(self.device)
    
    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)
    reward = br.copy().reshape((self.params.batch_size,-1))
    reward = torch.FloatTensor(reward).to(self.device)

    target_q = reward
    if not self.params.force_term:
      current_q = self.critic_goal_only(state, task_vec, action)
    else:
      current_q = self.critic(state, task_vec, action)

    critic_loss = F.mse_loss(current_q, target_q)

    self.params.writer.add_scalar('train_critic/critic_loss',critic_loss, self.step)
    self.params.writer.add_scalar('train_critic/reward',reward.mean(), self.step)
    self.params.writer.add_scalar('train_critic/current_Q',current_q.mean(), self.step)
    self.params.writer.add_scalar('train_critic/reward_max',reward.max(), self.step)
    self.params.writer.add_scalar('train_critic/current_Q_max',current_q.max(), self.step)
     
    # optimize the critic
    if not self.params.force_term:
      self.critic_goal_only_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_goal_only_optimizer.step()
    else:
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

    if self.step % self.policy_freq == 0:
      print("updating actor")
      if not self.params.force_term:
        actor_loss = -self.critic_goal_only(state, task_vec, self.actor(state, task_vec, training=True)[:, :self.params.a_dim]).mean()
        self.params.writer.add_scalar('train_actor/actor_Q',-actor_loss, self.step)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

      else:
        goal_pred, force_pred = self.actor(state, task_vec, training=False)
        goal_loss = F.mse_loss(goal_pred, action_gt)
        with torch.no_grad():
          goal_pred_combine = goal_pred
        force_pred = force_pred.reshape((len(goal_pred), -1))
        action_pred = torch.cat([goal_pred_combine, force_pred], axis=-1)
        actor_loss = -self.critic(state, task_vec, action_pred).mean() + 0.01 * goal_loss
        self.params.writer.add_scalar('train_actor/actor_Q',-actor_loss, self.step)
        if self.params.force_term:
          self.params.writer.add_scalar('train_actor/goal_pred',goal_loss, self.step)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

  def lr_scheduler(self, optimizer, lr):
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    return optimizer

  def learn_feedback(self):
    self.step_feedback += 1

    if self.pointer_feedback > self.params.mem_feedback_capacity:
      indices_sample = np.random.choice(self.params.mem_feedback_capacity, size=self.params.batch_size * 2)
    else:
      indices_sample = np.random.choice(self.pointer_feedback, size=self.params.batch_size * 2)

    indices = []
    for indi in indices_sample:
      if self.memory_feedback[indi, 0] > 0.5:
        indices.append(indi)
      if len(indices) == self.params.batch_size:
        break
    indices = np.array(indices)

    bt = self.memory_feedback[indices, :]

    index1 = 1
    index2 = index1 + self.params.state_dim * self.params.stack_num
    bs = bt[:, index1: index2]

    index3 = index2 + self.params.a_dim
    ba = bt[:, index2: index3]

    index4 = index3 + 1
    br = bt[:, index3: index4]

    index5 = index4 + 1
    bd = bt[:, index4: index5]

    index6 = index5 + self.params.task_dim
    btask = bt[:, index5: index6]

    indices_next = [(indi + 1) % self.params.mem_feedback_capacity for indi in indices]
    indices_next = np.array(indices_next)

    bs_next = self.memory_feedback[indices_next, :][:, index1: index2]

    state = bs.copy().reshape((-1, 120, 160, 3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    action = ba.copy().reshape((-1, self.params.a_dim))
    action = torch.FloatTensor(action).to(self.device)

    state_next = bs_next.copy().reshape((-1, 120, 160, 3)).astype(np.uint8)
    state_next_list = [transforms(s) for s in state_next]
    state_next = torch.stack(state_next_list)
    state_next = torch.FloatTensor(state_next).to(self.device)

    task_vec = btask.copy().reshape((-1, self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)

    reward = br.copy().reshape((-1, 1))
    reward = torch.FloatTensor(reward).to(self.device)

    with torch.no_grad():
       action_next = self.actor_feedback_target(state_next, task_vec)
       q_next = self.critic_feedback_target(state_next, task_vec, action_next)
       not_done = torch.FloatTensor(1.0 - bd.astype(np.float32)).to(self.device)
       target_q = reward + not_done * self.params.discount * q_next

    current_q = self.critic_feedback(state, task_vec, action)
    critic_loss = F.mse_loss(current_q, target_q)

    # optimize the critic_feedback
    self.critic_feedback_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_feedback_optimizer.step()

    self.params.writer.add_scalar('train_critic_feedback_loss/critic_loss', critic_loss, self.step_feedback)
    self.params.writer.add_scalar('train_critic_feedback_reward/reward', reward.mean(), self.step_feedback)
    self.params.writer.add_scalar('train_critic_feedback/current_Q', current_q.mean(), self.step_feedback)
    self.params.writer.add_scalar('train_critic_feedback/reward_max', reward.max(), self.step_feedback)
    self.params.writer.add_scalar('train_critic_feedback/current_Q_max', current_q.max(), self.step_feedback)

    if 1:
      print("updating actor_feedback")
      #lr_tmp = self.params.a_f_lr / (float(critic_loss2)*10 + 1.0)
      #self.actor_optimizer = self.lr_scheduler(self.actor_optimizer, lr_tmp)
      actor_loss = -self.critic_feedback(state, task_vec, self.actor_feedback(state, task_vec)).mean()
      self.params.writer.add_scalar('train_actor_feedback/actor_Q', -actor_loss, self.step_feedback)

      # Optimize the actor
      self.actor_feedback_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_feedback_optimizer.step()

    for param, target_param in zip(self.critic_feedback.parameters(), self.critic_feedback_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    for param, target_param in zip(self.actor_feedback.parameters(), self.actor_feedback_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

  def save_model_actor_critic(self, step):
    save_path_critic = os.path.join(self.params.model_dir, 'critic_'+str(step)+'_model.pth.tar')
    torch.save(self.critic.state_dict(), save_path_critic)

    save_path_actor = os.path.join(self.params.model_dir, 'actor_'+str(step)+'_model.pth.tar')
    torch.save(self.actor.state_dict(), save_path_actor)

  def restore_actor(self, step, restore_path=None):
    if self.params.restore_old:
      self.restore_actor_old(step, restore_path)
    else:
      if restore_path is None:
        restore_path = self.params.model_dir
      print("restore from ",restore_path," at step ", step)
      save_path_actor = os.path.join(restore_path, 'actor_'+str(step)+'_model.pth.tar')
      self.actor.load_state_dict(torch.load(save_path_actor))

  def restore_force(self, step, restore_path):
    print("restore from ", restore_path, " at step ", step)
    save_path_actor = os.path.join(restore_path, 'actor_' + str(step) + '_model.pth.tar')
    self.actor.load_state_dict(torch.load(save_path_actor))

  def restore_actor_old(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_actor = os.path.join(restore_path, 'actor_'+str(step)+'_model.pth.tar')
    state_dict = torch.load(save_path_actor)
    own_state = self.actor.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        continue
      param = param.data
      print("name", name)
      print("own_state", own_state[name].size())
      print("state_dict", state_dict[name].size())
      if state_dict[name].size() == own_state[name].size():
        own_state[name].copy_(param)
    #self.actor.load_state_dict(torch.load(save_path_actor))

  def restore_actor_goal_only(self, step, restore_path=None):
    if self.params.restore_old:
      self.restore_actor_old(step, restore_path)
    else:
      if restore_path is None:
        restore_path = self.params.model_dir
      print("restore from ",restore_path," at step ", step)
      save_path_actor = os.path.join(restore_path, 'actor_'+str(step)+'_model.pth.tar')
      self.actor_goal_only.load_state_dict(torch.load(save_path_actor))

  def restore_actor_critic(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_critic = os.path.join(restore_path, 'critic_'+str(step)+'_model.pth.tar')
    self.critic.load_state_dict(torch.load(save_path_critic))

    save_path_actor = os.path.join(restore_path, 'actor_'+str(step)+'_model.pth.tar')
    self.actor.load_state_dict(torch.load(save_path_actor))

  def save_model_feedback(self, step):
    save_path_critic_f = os.path.join(self.params.model_dir, 'critic_feedback_'+str(step)+'_model.pth.tar')
    torch.save(self.critic_feedback.state_dict(), save_path_critic_f)

    save_path_actor_f = os.path.join(self.params.model_dir, 'actor_feedback_'+str(step)+'_model.pth.tar')
    torch.save(self.actor_feedback.state_dict(), save_path_actor_f)

  def restore_feedback(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_critic_f = os.path.join(restore_path, 'critic_feedback_'+str(step)+'_model.pth.tar')
    self.critic_feedback.load_state_dict(torch.load(save_path_critic_f))

    save_path_actor_f = os.path.join(restore_path, 'actor_feedback_'+str(step)+'_model.pth.tar')
    self.actor_feedback.load_state_dict(torch.load(save_path_actor_f))

  def save_model_master(self, step):
    save_path_master = os.path.join(self.params.model_dir, 'master_'+str(step)+'_model.pth.tar')
    torch.save(self.master.state_dict(), save_path_master)

  def restore_master(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_master = os.path.join(restore_path, 'master_'+str(step)+'_model.pth.tar')
    self.master.load_state_dict(torch.load(save_path_master))

  def save_model_master_feedback(self, step):
    save_path_master_feedback = os.path.join(self.params.model_dir, 'master_feedback_'+str(step)+'_model.pth.tar')
    torch.save(self.master_feedback.state_dict(), save_path_master_feedback)

  def restore_master_feedback(self, step, restore_path=None):
    if restore_path is None:
      restore_path = self.params.model_dir
    print("restore from ",restore_path," at step ", step)
    save_path_master_feedback = os.path.join(restore_path, 'master_feedback_'+str(step)+'_model.pth.tar')
    self.master_feedback.load_state_dict(torch.load(save_path_master_feedback))

  def imitate_learn(self, imitate_memory):
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

    gt_force = []
    gt_goal = []

    bgoal = ba[:, :self.params.a_dim]
    bforce = ba[:, self.params.a_dim:].reshape((self.params.batch_size, self.params.traj_timesteps, self.params.a_dim))

    for idd in range(self.params.batch_size):
      gt_imitate = DMP(self.params, n_dmps=self.params.a_dim, goal=b_traj_start[idd] + bgoal[idd], start=b_traj_start[idd], force=bforce[idd], timesteps=self.params.traj_timesteps)
      gt_path = gt_imitate.rollout()[0]
      gt_goal_single = gt_path[-1] - gt_path[0]
      dmp_cal = DMP(self.params, n_dmps=self.params.a_dim, goal=gt_path[-1], start=gt_path[0],
                     force=np.zeros((self.params.traj_timesteps, self.params.a_dim)), timesteps=self.params.traj_timesteps)
      gt_force_single = dmp_cal.imitate_path(gt_path)
      gt_force_single = np.array(gt_force_single)
      action_max = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(self.params.traj_timesteps)
      action_max = np.tile(action_max, [1, self.params.a_dim])
      gt_force_single = (gt_force_single).clip(-1.0 * action_max, action_max)
      gt_force.append(gt_force_single)
      gt_goal.append(gt_goal_single)

      if 0:
        dmp_cal = DMP(self.params, n_dmps=self.params.a_dim, goal=gt_path[0] + gt_goal_single, start=gt_path[0],
                      force=gt_force_single,
                      timesteps=self.params.traj_timesteps)
        cal_path = dmp_cal.rollout()[0]
        fig = plt.figure(1)
        lenT = self.params.traj_timesteps
        plt.plot(np.arange(0, lenT), gt_path[:, 1], color='red')
        plt.plot(np.arange(0, lenT), cal_path[:, 1] + 0.1, color='green')
        #plt.plot(np.arange(0, lenT), gt_force_single[:, 1], color='blue')
        fig.canvas.draw()
        images = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3)
        plt.show()

    gt_goal = np.array(gt_goal)
    goal_gt = torch.FloatTensor(gt_goal).to(self.device)
    gt_force = np.array(gt_force)
    force_gt = torch.FloatTensor(gt_force).to(self.device)

    state = bs.copy().reshape((-1,120,160,3)).astype(np.uint8)
    state_list = [transforms(s) for s in state]
    state = torch.stack(state_list)
    state = torch.FloatTensor(state).to(self.device)

    task_vec = btask.copy().reshape((-1,self.params.task_dim))
    task_vec = torch.FloatTensor(task_vec).to(self.device)

    for ik in range(1):
      self.imitate_step += 1
      offset = 0
      goal_pred, force_pred = self.master(state, task_vec)

      imitate_goal_loss = F.mse_loss(goal_gt, goal_pred)

      imitate_force_loss = F.mse_loss(force_gt, force_pred)

      imitate_loss = imitate_goal_loss + 1 * imitate_force_loss

      print("imitation ", self.imitate_step, "goal error", imitate_goal_loss, "force_error", imitate_force_loss)

      self.master_optimizer.zero_grad()
      imitate_loss.backward()
      self.master_optimizer.step()

      self.params.writer.add_scalar('train_imitate_loss/imitate_loss',imitate_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_goal_loss', imitate_goal_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate_loss/imitate_force_loss', imitate_force_loss, self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_0', goal_pred[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_1', goal_pred[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_2', goal_pred[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_3', goal_pred[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_4', goal_pred[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_5', goal_pred[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/pred_6', goal_pred[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_0', goal_gt[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_1', goal_gt[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_2', goal_gt[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_3', goal_gt[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_4', goal_gt[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_5', goal_gt[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('train_imitate_goal/gt_6', goal_gt[:,6].mean(),self.imitate_step)

      if 1:
        self.params.writer.add_scalar('train_imitate_force/gt_0', force_gt[:,:,0].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_1', force_gt[:,:,1].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_2', force_gt[:,:,2].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_3', force_gt[:,:,3].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_4', force_gt[:,:,4].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_5', force_gt[:,:,5].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/gt_6', force_gt[:,:,6].mean(), self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_0', force_pred[:,:,0].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_1', force_pred[:,:,1].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_2', force_pred[:,:,2].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_3', force_pred[:,:,3].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_4', force_pred[:,:,4].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_5', force_pred[:,:,5].mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_force/pred_6', force_pred[:,:,6].mean(), self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_0',(goal_gt[:,0]-goal_pred[:,0]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_1',(goal_gt[:,1]-goal_pred[:,1]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_2',(goal_gt[:,2]-goal_pred[:,2]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_3',(goal_gt[:,3]-goal_pred[:,3]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_4',(goal_gt[:,4]-goal_pred[:,4]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_5',(goal_gt[:,5]-goal_pred[:,5]).mean(),self.imitate_step)
        self.params.writer.add_scalar('train_imitate_goal/pred_error_6',(goal_gt[:,6]-goal_pred[:,6]).mean(),self.imitate_step)


  def imitate_learn_test(self, imitate_memory):
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

    bgoal = ba[:, :self.params.a_dim]
    bforce = ba[:, self.params.a_dim:].reshape((self.params.batch_size, self.params.traj_timesteps, self.params.a_dim))

    gt_force = []
    gt_goal = []
    for idd in range(self.params.batch_size):
      gt_imitate = DMP(self.params, n_dmps=self.params.a_dim, goal=b_traj_start[idd] + bgoal[idd], start=b_traj_start[idd], force=bforce[idd], timesteps=self.params.traj_timesteps)
      gt_path = gt_imitate.rollout()[0]
      gt_goal_single = gt_path[-1] - gt_path[0]
      dmp_cal = DMP(self.params, n_dmps=self.params.a_dim, goal=gt_path[0] + gt_goal_single, start=gt_path[0],
                     force=np.zeros((self.params.traj_timesteps, self.params.a_dim)), timesteps=self.params.traj_timesteps)
      gt_force_single = dmp_cal.imitate_path(gt_path)
      action_max = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(
        self.params.traj_timesteps)
      action_max = np.tile(action_max, [1, self.params.a_dim])
      gt_force_single = gt_force_single.clip(-action_max, action_max)
      gt_force.append(gt_force_single)
      gt_goal.append(gt_goal_single)

    gt_goal = np.array(gt_goal)
    goal_gt = torch.FloatTensor(gt_goal).to(self.device)
    gt_force = np.array(gt_force)
    force_gt = torch.FloatTensor(gt_force).to(self.device)

    for ik in range(1):
      offset = 0
      goal_pred, force_pred = self.master(state, task_vec)

      imitate_goal_loss = F.mse_loss(goal_gt, goal_pred)
      imitate_force_loss = F.mse_loss(force_gt, force_pred)
      imitate_loss = imitate_goal_loss + 1 * imitate_force_loss

      self.params.writer.add_scalar('test_imitate_loss/imitate_loss',imitate_loss, self.imitate_step)
      self.params.writer.add_scalar('test_imitate_loss/imitate_goal_loss', imitate_goal_loss, self.imitate_step)
      self.params.writer.add_scalar('test_imitate_loss/imitate_force_loss', imitate_force_loss, self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_0', goal_pred[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_1', goal_pred[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_2', goal_pred[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_3', goal_pred[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_4', goal_pred[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_5', goal_pred[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/pred_6', goal_pred[:,6].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_0', goal_gt[:,0].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_1', goal_gt[:,1].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_2', goal_gt[:,2].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_3', goal_gt[:,3].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_4', goal_gt[:,4].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_5', goal_gt[:,5].mean(),self.imitate_step)
      self.params.writer.add_scalar('test_imitate_goal/gt_6', goal_gt[:,6].mean(),self.imitate_step)

      if 1:
        self.params.writer.add_scalar('test_imitate_force/gt_0', force_gt[:,:,0].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_1', force_gt[:,:,1].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_2', force_gt[:,:,2].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_3', force_gt[:,:,3].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_4', force_gt[:,:,4].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_5', force_gt[:,:,5].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/gt_6', force_gt[:,:,6].mean(), self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_0', force_pred[:,:,0].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_1', force_pred[:,:,1].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_2', force_pred[:,:,2].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_3', force_pred[:,:,3].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_4', force_pred[:,:,4].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_5', force_pred[:,:,5].mean(),self.imitate_step)
        self.params.writer.add_scalar('test_imitate_force/pred_6', force_pred[:,:,6].mean(), self.imitate_step)
        #print('test_imitate_force_0/pred_error_action_0',(gt_action_label[:,0]- goal_pred[:,0]).mean(),self.imitate_step)
        #print('test_imitate_force_1/pred_error_action_1',(gt_action_label[:,1]- goal_pred[:,1]).mean(),self.imitate_step)
        #print('test_imitate_force_2/pred_error_action_2',(gt_action_label[:,2]- goal_pred[:,2]).mean(),self.imitate_step)
        #print('test_imitate_force_3/pred_error_action_3',(gt_action_label[:,3]- goal_pred[:,3]).mean(),self.imitate_step)
        #print('test_imitate_force_4/pred_error_action_4',(gt_action_label[:,4]- goal_pred[:,4]).mean(),self.imitate_step)
        #print('test_imitate_force_5/pred_error_action_5',(gt_action_label[:,5]- goal_pred[:,5]).mean(),self.imitate_step)
        #print('test_imitate_force_6/pred_error_action_6',(gt_action_label[:,6]- goal_pred[:,6]).mean(),self.imitate_step)
    return imitate_goal_loss.cpu().detach().numpy(), imitate_force_loss.cpu().detach().numpy(), imitate_loss.cpu().detach().numpy()
