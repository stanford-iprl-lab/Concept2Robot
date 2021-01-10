import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np
from collections import deque
import time
np.set_printoptions(precision=4,suppress=False)

import importlib
import imageio
import math
from tensorboardX import SummaryWriter
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DMP_DIR = os.path.join(BASE_DIR,'../deepTraj')
PROJECT_DIR = os.path.join(BASE_DIR,'../')
sys.path.insert(0,DMP_DIR)
from ddmp import DDMP  as DMP

sys.path.insert(0, "../simulation")
sys.path.insert(0,"../external/bullet3/build_cmake/examples/pybullet")
import pybullet

import bullet_client as bc

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from scipy.special import softmax

#####
from agents import Agent

class Enver(object):
  def __init__(self,task_id,params):
    self.params = params
    self.TaskId = task_id
    self.wid = task_id
    self.p_id = bc.BulletClient(connection_mode=pybullet.GUI)
    time_sleep = np.random.uniform(0,100)
    env_module = importlib.import_module("env_{}".format(self.TaskId))
    RobotEnv = getattr(env_module, "Engine{}".format(self.TaskId))
    self.env = RobotEnv(worker_id=self.wid, opti=self.params, cReward=self.params.video_reward, p_id=self.p_id,
                            taskId=self.TaskId, n_dmps=7)


class Worker(object):
    def __init__(self, params):
        self.params = params

        if self.params.stage == 'imitation':
            self.file_name = '{}/{}/{}'.format("imitation", str(self.params.exp_name), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == 'test':
            self.file_name = '{}_{}/{}'.format(str(self.params.task_id), "test", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == 'train':
            self.file_name = '{}_{}/{}_{}'.format(str(self.params.task_id), str(self.params.exp_name), str(self.params.method),
                                                  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        elif self.params.stage == "demonstration":
            self.model_id = self.params.restore_path.strip().split('/')[-1]
            self.file_name = '{}/{}/{}'.format(str(self.params.task_id),str(self.params.exp_name),str(self.model_id))
        elif self.params.stage == 'feedback_train':
            self.file_name = '{}_{}_feedback_train/{}_{}'.format(str(self.params.task_id), str(self.params.exp_name), str(self.params.method),
                                                  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        elif self.params.stage == 'feedback_test':
            self.file_name = '{}_{}/{}'.format(str(self.params.task_id), "feedback_test",
                                               datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == "demonstration_feedback":
            self.model_id = self.params.restore_path.strip().split('/')[-1]
            self.file_name = '{}/{}/{}'.format(str(self.params.task_id),str(self.params.exp_name),str(self.model_id))
        elif self.params.stage == 'imitation_feedback':
            self.file_name = '{}/{}/{}'.format("imitation_feedback", str(self.params.exp_name), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        else:
            self.file_name = None

        ### model_dir, writer and log_dir
        if self.params.stage == 'train' or self.params.stage == 'imitation' or self.params.stage == "feedback_train" or self.params.stage == 'imitation_feedback':
            self.writer = SummaryWriter(logdir=os.path.join(self.params.log_dir, self.file_name))
            self.model_dir = os.path.join(self.params.save_dir, self.file_name)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        elif self.params.stage == 'test' or self.params.stage == 'feedback_test':
            self.model_dir = None
            self.writer = None
        elif self.params.stage == "demonstration" or self.params.stage == "demonstration_feedback":
            self.model_dir = None
            self.writer = None
            self.saving_data_dir = os.path.join(self.params.demonstration_dir, self.file_name)
            if not os.path.exists(self.saving_data_dir):
                os.makedirs(self.saving_data_dir)
            print("saving_data_dir",self.saving_data_dir)
        else:
            self.model_dir = None
            self.writer = None

        self.log_dir = os.path.join(self.params.log_dir, self.file_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.params.writer = self.writer
        self.params.model_dir = self.model_dir

        self.wid = params.wid

        self.agent = Agent(self.params)
        time_sleep = np.random.uniform(0, 100)
        print("time_sleep", time_sleep)

        if self.params.gui:
            self.p_id = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)


        self.TaskId = self.params.task_id
        if (self.params.stage == "imitation"):
           print("imitation!!!!")
           self.env_list = {}
           self.p_id_list = {}
           self.env = None
        else:
           print("worker id %d" % self.wid, "task id", self.TaskId)
           env_module = importlib.import_module("env_{}".format(self.TaskId))
           RobotEnv = getattr(env_module, "Engine{}".format(self.TaskId))
           self.env = RobotEnv(worker_id=self.wid, opti=self.params, cReward=self.params.video_reward, p_id=self.p_id,
                            taskId=self.TaskId, n_dmps=7)
           self.task_vec = np.load("../Languages/" + str(self.TaskId) + '.npy')


    def train(self, restore_episode=0, restore_path=None):
        explore_var = self.params.explore_var  # control exploratin
        episode_num = 0
        suc_num = 0
        step_check = 0
        reward_check = 0

        if restore_path is not None:
            self.agent.restore(restore_episode, restore_path)

        total_episode = 0 + restore_episode

        if total_episode > 0:
            explore_var = explore_var * self.params.explore_decay ** (total_episode)

        print("total_episode", total_episode, "var", explore_var)

        for ep_iter in range(1, self.params.max_ep):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()
            while True:
                # First generate action
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))

                if total_episode < self.params.start_learning_episode:
                    action_pred = np.random.uniform(-0.5, 0.5, size=(self.params.traj_timesteps + 1) * self.params.a_dim)
                else:
                    if not self.params.force_term:
                        action_pred[:self.params.a_dim] = self.agent.choose_action(observation, self.task_vec)
                    else:
                        action_pred = self.agent.choose_action(observation, self.task_vec)
                action_clip = np.copy(action_pred)

                # Add exploration
                explore_var = max(explore_var, 0.1)
                action_clip = np.random.normal(action_clip, explore_var)
                self.writer.add_scalar("train/explore_var", explore_var, ep_iter)

                # Clip the goal in action_clip
                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action,
                                                               self.params.rotation_max_action)
                print("action_clip_rotation", action_clip[3:6])

                action_goal = np.zeros((self.params.a_dim,))
                action_goal = action_clip[:self.params.a_dim]

                if self.params.force_term:
                    # clip the forces in action_clip
                    action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))
                    forces_pred = action_clip[7:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = forces_pred * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)

                    ### force term in action clip
                    forces_temp = np.copy(forces_pred) * weights
                    forces_temp[:, 3:6] = (forces_temp[:, 3:6]).clip( -self.params.rotation_max_action / np.pi * 50.0,
                        self.params.rotation_max_action / np.pi * 50.0)
                    forces_temp[:-1, :] = forces_temp[:-1, :] / weights[:-1,
                                                                :] / 50.0  #### the last element of weight is zero.
                    forces_temp = forces_temp.transpose().reshape((-1,))
                    action_clip[self.params.a_dim:] = forces_temp

                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                reset_flag = False
                observation_c = np.copy(observation_next).reshape((-1,))

                # keep iteration until done
                while not done:
                    feedback_flag = None
                    action_null = np.zeros((self.params.a_dim,))
                    observation_next, reward, done, suc = self.env.step(action_null, None, feedback_flag, reset_flag)
                    if not done:
                        reward = 0

                print("gt_reward", self.params.gt_reward)
                if self.params.gt_reward:
                    reward = float(suc) * 3.0

                self.writer.add_scalar("train/reward", reward, ep_iter)
                if not self.params.force_term:
                    action_penalty = np.sum(np.abs(action_clip[:self.params.a_dim])) / float(
                        self.params.a_dim) * self.params.action_penalty
                else:
                    action_penalty = np.sum(np.abs(action_clip[:self.params.a_dim])) / float(
                        self.params.a_dim) * self.params.action_penalty + np.sum(
                        np.abs(action_clip[self.params.a_dim:])) / float(
                        self.params.a_dim * self.params.traj_timesteps) * self.params.action_penalty * 0.1

                reward = reward - action_penalty

                print("Training", "success", suc, "taskid", self.TaskId, "ep_iter", ep_iter, " action", action_clip[:7], "action_pred", action_pred[:7],
                      "reward", reward, "suc", suc, "action_penalty", action_penalty)

                self.writer.add_scalar("train_action_0/pred_action", action_pred[0], ep_iter)
                self.writer.add_scalar("train_action_0/clip_action", action_clip[0], ep_iter)
                self.writer.add_scalar("train_action_1/pred_action", action_pred[1], ep_iter)
                self.writer.add_scalar("train_action_1/clip_action", action_clip[1], ep_iter)
                self.writer.add_scalar("train_action_2/pred_action", action_pred[2], ep_iter)
                self.writer.add_scalar("train_action_2/clip_action", action_clip[2], ep_iter)
                self.writer.add_scalar("train_action_6/pred_action", action_pred[6], ep_iter)
                self.writer.add_scalar("train_action_6/clip_action", action_clip[6], ep_iter)

                observation_next = np.reshape(observation_next, (-1,))

                # traj_real is different from traj generated
                while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                    self.env.real_traj_list.append(self.env.real_traj_list[-1])
                traj_real = np.array(self.env.real_traj_list)
                traj_real = traj_real.reshape((-1,))

                step_check += 1
                reward_check += reward

                # Important!! the shape of action_pred is ((self.params.traj_timesteps + 1) * self.params.a_dim)
                self.agent.store_transition(initial_pose, traj_real, observation, action_clip, reward, done, self.task_vec)

                if ep_iter > self.params.start_learning_episode:
                    explore_var = explore_var * self.params.explore_decay  # decay the action randomness
                    a = self.agent.learn()
                    print("starting to learn")

                observation = observation_next
                total_episode += 1

                if ep_iter % self.params.saving_model_freq == 0:
                    current_performance = float(suc_num) / (episode_num + 0.0001)
                    print("suc rate %f reward %f var %f" % (current_performance, float(reward_check) / (episode_num + 0.0001), explore_var))
                    print("saving models at step%d" % ep_iter)
                    self.agent.save_model_actor_critic(ep_iter)
                    self.test(ep_iter)

                if done:
                    self.writer.add_scalar("train/success_rate", suc, ep_iter)
                    suc_num += suc
                    episode_num += 1
                    break

    def test(self, restore_episode=0, restore_path=None):
        total_suc = 0

        if self.params.stage == "imitation":
            self.p_id.__del__()
            del self.env
            self.env = Enver(self.TaskId, self.params).env

        if restore_path is not None:
            if self.params.stage == "imitation":
                self.agent.restore_master(restore_episode, restore_path)
            else:
                self.agent.restore_actor_critic(restore_episode, restore_path)
            print("testing from restoring sth", restore_path)

        if self.params.stage == "demonstration":
            max_iteration = self.params.max_ep_demon
        elif self.params.stage == "test" or self.params.stage == "train":
            max_iteration = self.params.max_ep_test
        elif self.params.stage == "imitation":
            max_iteration = self.params.max_ep_imitation
            self.task_vec = np.load("../Languages/"+str(self.TaskId)+'.npy')

        for ep_iter in range(max_iteration):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))
            observation_init = np.copy(observation)

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()

            while True:
                # First generate action
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))

                if self.params.stage == "imitation":
                    action_pred = self.agent.choose_action_master(observation, self.task_vec)
                else:
                    if self.params.force_term:
                        action_pred = self.agent.choose_action(observation, self.task_vec)
                    else:
                        action_pred[:self.params.a_dim] = self.agent.choose_action(observation, self.task_vec)
                action_clip = np.copy(action_pred)

                # Clip the goal in action_clip
                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
                action_goal = np.zeros((7,))
                action_goal[:7] = action_clip[:7]

                if self.params.force_term or self.params.stage == "imitation":
                    #clip the forces in action_clip
                    action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))
                    forces_pred = action_clip[self.params.a_dim:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = forces_pred * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)
                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                reset_flag = False
                observation_c = np.copy(observation_next).reshape((-1,))

                while not done:
                    feedback_term = np.zeros((7,))
                    observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_term, reset_flag)

                print("Tesing", "success", suc, "taskid", self.TaskId, "ep_iter", ep_iter, " action", action_clip[:7], "action_pred", action_pred[:7],
                      "reward", reward, "suc", suc, "dmp dist",
                      np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]),
                      self.env.robotCurrentStatus()[:3], self.env.dmp.goal[:3])

                observation = observation_next

                if suc and self.params.stage == "demonstration":
                    # generate trajectory
                    while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                        self.env.real_traj_list.append(self.env.real_traj_list[-1])
                    traj_real = np.array(self.env.real_traj_list)
                    traj_real = traj_real.reshape((-1,))

                    # Clip the goal in action_pred
                    action_pred[:3] = action_pred[:3].clip(-self.params.max_action, self.params.max_action)
                    action_pred[3:6] = (action_pred[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action,
                                                                             self.params.rotation_max_action)

                    # Clip the force in action_pred
                    forces_temp = action_pred[self.params.a_dim:].reshape(
                        (self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    forces_temp = forces_temp * weights
                    forces_temp[:, 3:6] = (forces_temp[:, 3:6]).clip( -self.params.rotation_max_action / np.pi * 50.0,
                        self.params.rotation_max_action / np.pi * 50.0)
                    forces_temp[:-1, :] = forces_temp[:-1, :] / weights[:-1, :] / 50.0 #### the last element of weight is zero.
                    forces_temp = forces_temp.transpose().reshape((-1,))
                    action_pred[self.params.a_dim:] = forces_temp
                    # Imporant!!! action_pred is clipped and has shape of (self.params.traj_timesteps + 1) * self.params.a_dim
                    self.agent.store_transition_locally(initial_pose, traj_real, observation_init, action_pred, reward,
                                                        done, self.task_vec, self.saving_data_dir)

                if done:
                    total_suc += float(suc)
                    ### whether to generated gif
                    recordGif = self.params.recordGif
                    if recordGif:
                        classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
                        recordGif_dir = os.path.join(self.params.gif_dir, str(self.params.task_id))
                        if not os.path.exists(recordGif_dir):
                            os.makedirs(recordGif_dir)
                        imageio.mimsave(os.path.join(recordGif_dir, str(self.params.task_id) + '_' + str(ep_) + '.gif'),
                                        self.env.obs_list)
                    break

        perf = total_suc / float(max_iteration)
        print("success performance", perf)

        if not self.params.stage == 'imitation':
            np.savetxt(os.path.join(self.log_dir,'successRate.txt'), np.array([perf]), fmt='%1.4f')

        print("successfully done!!!!!")
        return perf


    def feedback(self, restore_episode=0, restore_path=None, restore_timestep=0, restore_feedback_path=None):
        total_suc = 0

        explore_var = self.params.explore_var  # control exploratin

        if restore_path is not None:
            self.agent.restore_actor_critic(restore_episode, restore_path)
            print("testing from restoring sth", restore_path)

        if restore_feedback_path is not None:
            self.agent.restore_feedback(restore_timestep, restore_feedback_path)
            print("testing from restoring sth", restore_path)

        total_timestep = 0

        for ep_iter in range(self.params.max_ep):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))
            observation_init = np.copy(observation)

            if self.params.debug:
                prev_pos = self.env.pos[1]
            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()

            while True:
                # First generate action
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))

                if self.params.force_term:
                    action_pred = self.agent.choose_action(observation, self.task_vec)
                else:
                    action_pred[:self.params.a_dim] = self.agent.choose_action(observation, self.task_vec)

                if not self.params.debug:
                    action_clip = np.copy(action_pred)

                # Clip the goal in action_clip
                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
                action_goal = np.zeros((7,))
                action_goal[:7] = action_clip[:7]

                if self.params.force_term:
                    #clip the forces in action_clip
                    action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))
                    forces_pred = action_clip[self.params.a_dim:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = forces_pred * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)
                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                ###############################################################################################
                ###############################################################################################
                reset_flag = False
                observation_feedback = np.copy(observation_next).reshape((-1,))

                if np.random.uniform() < 0.1:
                    exploration_flag = True
                else:
                    exploration_flag = False
                timestep_in_ep = 0
                stack_num = 4
                observation_feedback_list = deque(maxlen=self.params.stack_num)
                observation_feedback_list.append(observation_feedback)
                observation_feedback_next_list = deque(maxlen=self.params.stack_num)
                feedback_clip = np.zeros((self.params.a_dim,))
                feedback_pred = np.zeros((self.params.a_dim,))
                for i in range(self.params.stack_num-1):
                    if not done:
                        observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_clip, reset_flag)
                        observation_feedback = np.copy(observation_next).reshape((-1,))
                        observation_feedback_list.append(observation_feedback)
                    else:
                        observation_feedback_list.append(observation_feedback)

                observation_feedback_array = np.array(observation_feedback_list).reshape((-1,))
                while not done:
                    print("exploration",exploration_flag)
                    ##### feedback
                    if total_timestep < self.params.start_learning_timestep or exploration_flag:
                        feedback_pred = np.random.uniform(-self.params.max_feedback_action, self.params.max_feedback_action,
                                                        size=(self.params.a_dim))
                    else:
                        print(observation_feedback_array.shape)
                        feedback_pred = self.agent.choose_action_feedback(observation_feedback_array, self.task_vec)

                    feedback_clip = np.copy(feedback_pred)
                    explore_var = max(explore_var, 0.25)
                    feedback_clip += np.random.normal(0, self.params.max_feedback_action * explore_var, size=self.params.a_dim)

                    feedback_clip[:3] = (feedback_clip[:3]).clip(
                        -self.params.max_feedback_action, self.params.max_feedback_action)
                    feedback_clip[3:6] = (feedback_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_feedback_action, self.params.rotation_max_feedback_action)

                    reward_acc = 0
                    for i in range(self.params.stack_num):
                        if not done:
                            observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_clip, reset_flag)
                            reward_acc += reward
                            observation_feedback_next = np.copy(observation_next).reshape((-1,))
                            observation_feedback_next_list.append(observation_feedback_next)
                        else:
                            observation_feedback_next_list.append(observation_feedback_next)

                    observation_feedback_next_array = np.array(observation_feedback_next_list).reshape((-1,))

                    self.writer.add_scalar("feedback_train_0_clip/action_0", feedback_clip[0], total_timestep)
                    self.writer.add_scalar("feedback_train_1_clip/action_1", feedback_clip[1], total_timestep)
                    self.writer.add_scalar("feedback_train_2_clip/action_2", feedback_clip[2], total_timestep)

                    self.writer.add_scalar("feedback_train_0/action_0", feedback_pred[0], total_timestep)
                    self.writer.add_scalar("feedback_train_1/action_1", feedback_pred[1], total_timestep)
                    self.writer.add_scalar("feedback_train_2/action_2", feedback_pred[2], total_timestep)

                    if self.params.debug:
                        if suc > 0:
                            reward = 5
                        if done and reward < 0:
                            reward = -2
                        if not done:
                            reward = 0
                        pos = np.array(self.env.p.getBasePositionAndOrientation(self.env.obj_id)[0])

                        reward += - 20 * np.linalg.norm(feedback_clip)
                        reward += (pos[1] - prev_pos) * 30
                        prev_pos = pos[1]
                    else:
                    #if suc > 0:
                        reward += - 2 * np.linalg.norm(feedback_clip)
                        if not suc:
                            if reward >= 0:
                                reward = 0

                    #      pos = np.array(self.env.p.getBasePositionAndOrientation(self.env.obj_id)[0])
                    #      reward = (pos[1] - prev_pos) * 0 - 2 * np.linalg.norm(feedback_clip)
                    #      prev_pos = pos[1]

                    self.writer.add_scalar("feedback_reward/reward", reward, total_timestep)
                    self.writer.add_scalar("feedback_action/action", np.linalg.norm(feedback_pred), total_timestep)
                    self.writer.add_scalar("feedback_explore/explore_var", explore_var, total_timestep)

                    total_timestep += 1

                    print("training feedback_pred",feedback_pred,"total_timestep",total_timestep,"done",done,"reward",reward,"suc",suc)

                    if self.params.stage == "feedback_train":
                        self.agent.store_transition_feedback(1, observation_feedback_array, feedback_clip, reward, done, self.task_vec)
                        if done:
                            self.agent.store_transition_feedback(0, observation_feedback_next_array, feedback_clip, reward, done, self.task_vec)

                    observation_feedback_array = observation_feedback_next_array

                    if total_timestep > self.params.start_learning_timestep:
                        explore_var = explore_var * self.params.explore_decay
                        self.agent.learn_feedback()
                    if done:
                        self.writer.add_scalar("feedback_train/success_rate", suc, ep_iter)

                observation = observation_next

                if ep_iter % self.params.saving_model_freq == 0:
                    current_performance = float(total_suc) / (ep_iter + 0.0001)
                    print("suc rate %f var %f" % (current_performance, explore_var))
                    print("saving models at step%d" % ep_iter)
                    self.agent.save_model_feedback(ep_iter)
                    perf = self.feedback_test()
                    self.writer.add_scalar("feedback_test/perf", perf, ep_iter)

                if done:
                    total_suc += float(suc)
                    ### whether to generated gif
                    recordGif = self.params.recordGif
                    if recordGif:
                        classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
                        recordGif_dir = os.path.join(self.params.gif_dir, str(self.params.task_id))
                        if not os.path.exists(recordGif_dir):
                            os.makedirs(recordGif_dir)
                        imageio.mimsave(os.path.join(recordGif_dir, str(self.params.task_id) + '_' + str(ep_) + '.gif'),
                                        self.env.obs_list)
                    break

        perf = total_suc / float(self.params.max_ep)
        print("success performance", perf)
        np.savetxt(os.path.join(self.log_dir,'successRate.txt'), np.array([perf]), fmt='%1.4f')

        print("successfully done!!!!!")

    def feedback_test(self, restore_episode=0, restore_path=None, restore_feedback_episode=0, restore_feedback_path=None):
        total_suc = 0

        if restore_path is not None:
            self.agent.restore_actor_critic(restore_episode, restore_path)
            print("testing from restoring sth", restore_path)

        if restore_feedback_path is not None:
            self.agent.restore_feedback(restore_feedback_episode, restore_feedback_path)
            print("testing from restoring sth", restore_path)

        if self.params.stage == "demonstration_feedback":
            max_iteration = self.params.max_ep_demon
        elif self.params.stage == "feedback_test" or self.params.stage == "feedback_train":
            max_iteration = self.params.max_ep_feedback_test

        for ep_iter in range(max_iteration):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))
            observation_init = np.copy(observation)
            timestep_in_ep = 0

            observation_list = []
            feedback_pred_list = []
            reward_list = []
            done_list = []

            if self.params.debug:
                prev_pos = self.env.pos[1]

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()

            while True:
                # First generate action
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))

                if self.params.force_term:
                    action_pred = self.agent.choose_action(observation, self.task_vec)
                else:
                    action_pred[:self.params.a_dim] = self.agent.choose_action(observation, self.task_vec)

                if not self.params.debug:
                    action_clip = np.copy(action_pred)

                # Clip the goal in action_clip
                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)
                action_goal = np.zeros((7,))
                action_goal[:7] = action_clip[:7]

                if self.params.force_term:
                    #clip the forces in action_clip
                    action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))
                    forces_pred = action_clip[self.params.a_dim:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = forces_pred * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)
                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                ###############################################################################################
                ###############################################################################################
                reset_flag = False
                observation_feedback = np.copy(observation_next).reshape((-1,))

                timestep_in_ep = 0
                stack_num = 4
                observation_feedback_list = deque(maxlen=self.params.stack_num)
                observation_feedback_list.append(observation_feedback)
                observation_feedback_next_list = deque(maxlen=self.params.stack_num)
                feedback_clip = np.zeros((self.params.a_dim,))
                feedback_pred = np.zeros((self.params.a_dim,))
                for i in range(self.params.stack_num-1):
                    if not done:
                        observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_clip, reset_flag)
                        observation_feedback = np.copy(observation_next).reshape((-1,))
                        observation_feedback_list.append(observation_feedback)
                    else:
                        observation_feedback_list.append(observation_feedback)

                observation_feedback_array = np.array(observation_feedback_list).reshape((-1,))
                while not done:
                    ##### feedback

                    feedback_pred = self.agent.choose_action_feedback(observation_feedback_array, self.task_vec)

                    feedback_clip = np.copy(feedback_pred)

                    feedback_clip[:3] = (feedback_clip[:3]).clip(
                        -self.params.max_feedback_action, self.params.max_feedback_action)
                    feedback_clip[3:6] = (feedback_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_feedback_action, self.params.rotation_max_feedback_action)

                    reward_acc = 0
                    for i in range(self.params.stack_num):
                        if not done:
                            observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_clip, reset_flag)
                            reward_acc += reward
                            observation_feedback_next = np.copy(observation_next).reshape((-1,))
                            observation_feedback_next_list.append(observation_feedback_next)
                        else:
                            observation_feedback_next_list.append(observation_feedback_next)

                    observation_feedback_next_array = np.array(observation_feedback_next_list).reshape((-1,))

                    if suc > 0:
                        reward = - 20 * np.linalg.norm(feedback_clip)
                    else:
                        if 1:#reward >= 0:
                        #    reward = 0
                          pos = np.array(self.env.p.getBasePositionAndOrientation(self.env.obj_id)[0])
                          reward = (pos[1] - prev_pos) * 0 - 20 * np.linalg.norm(feedback_clip)
                          prev_pos = pos[1]

                    print("test feedback_pred",feedback_pred,"episode ",ep_iter,"done",done,"reward",reward,"suc",suc)

                    if self.params.stage == "demonstration_feedback":
                        self.agent.store_transition_feedback_locally(timestep_in_ep, 1, observation_feedback_array, feedback_clip, reward, done, self.task_vec, self.saving_data_dir)
                        timestep_in_ep += 1

                    observation_feedback_array = observation_feedback_next_array

                observation = observation_next

                if suc and self.params.stage == "demonstration_feedback":
                    # generate trajectory
                    while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                        self.env.real_traj_list.append(self.env.real_traj_list[-1])
                    traj_real = np.array(self.env.real_traj_list)
                    traj_real = traj_real.reshape((-1,))

                    # Clip the goal in action_pred
                    action_pred[:3] = action_pred[:3].clip(-self.params.max_action, self.params.max_action)
                    action_pred[3:6] = (action_pred[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action,
                                                                             self.params.rotation_max_action)

                    # Clip the force in action_pred
                    forces_temp = action_pred[self.params.a_dim:].reshape(
                        (self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    forces_temp = forces_temp * weights
                    forces_temp[:, 3:6] = (forces_temp[:, 3:6]).clip( -self.params.rotation_max_action / np.pi * 50.0,
                        self.params.rotation_max_action / np.pi * 50.0)
                    forces_temp[:-1, :] = forces_temp[:-1, :] / weights[:-1, :] / 50.0 #### the last element of weight is zero.
                    forces_temp = forces_temp.transpose().reshape((-1,))
                    action_pred[self.params.a_dim:] = forces_temp
                    # Imporant!!! action_pred is clipped and has shape of (self.params.traj_timesteps + 1) * self.params.a_dim
                    self.agent.store_transition_locally(initial_pose, traj_real, observation_init, action_pred, reward,
                                                        done, self.task_vec, self.saving_data_dir)

                if done:
                    total_suc += float(suc)
                    ### whether to generated gif
                    recordGif = self.params.recordGif
                    if recordGif:
                        classes = [line.strip().split(":")[0] for line in open('../Languages/labels.txt')]
                        recordGif_dir = os.path.join(self.params.gif_dir, str(self.params.task_id))
                        if not os.path.exists(recordGif_dir):
                            os.makedirs(recordGif_dir)
                        imageio.mimsave(os.path.join(recordGif_dir, str(self.params.task_id) + '_' + str(ep_) + '.gif'),
                                        self.env.obs_list)
                    break

        perf = total_suc / float(self.params.max_ep_feedback_test)
        print("success performance", perf)
        np.savetxt(os.path.join(self.log_dir,'successRate.txt'), np.array([perf]), fmt='%1.4f')
        print("successfully done!!!!!")
        return perf

    def imitate(self):
        imitation_meta_info = "imitation_meta_info.txt"
        save_top_dir_dict = {}
        save_top_dir_dict[5] = "/juno/group/linshao/ConceptLearning/5" #"/scr1/Concept2Robot/5/with_force/rl_2021-01-04_18-11-50"
        save_top_dir_dict[13] = "/juno/group/linshao/ConceptLearning/13"
        save_top_dir_dict[14] = "/juno/group/linshao/ConceptLearning/14"
        save_top_dir_dict[15] = "/juno/group/linshao/ConceptLearning/15" #Yes
        save_top_dir_dict[20] = "/juno/group/linshao/ConceptLearning/20"
        save_top_dir_dict[40] = "/juno/group/linshao/ConceptLearning/40" # Yes
        save_top_dir_dict[41] = "/juno/group/linshao/ConceptLearning/41" # Yes
        save_top_dir_dict[42] = "/juno/group/linshao/ConceptLearning/42" # Yes
        save_top_dir_dict[43] = "/juno/group/linshao/ConceptLearning/43" # Yes
        save_top_dir_dict[44] = "/juno/group/linshao/ConceptLearning/44" # Yes but need to check the gripper
        save_top_dir_dict[45] = "/juno/group/linshao/ConceptLearning/45" # Yes
        save_top_dir_dict[46] = "/juno/group/linshao/ConceptLearning/46" # Yes
        save_top_dir_dict[47] = "/juno/group/linshao/ConceptLearning/47"
        save_top_dir_dict[86] = "/juno/group/linshao/ConceptLearning/86" # No
        save_top_dir_dict[87] = "/juno/group/linshao/ConceptLearning/87" # No
        save_top_dir_dict[88] = "/juno/group/linshao/ConceptLearning/88" # No
        save_top_dir_dict[89] = "/juno/group/linshao/ConceptLearning/89" # No
        save_top_dir_dict[93] = "/juno/group/linshao/ConceptLearning/93" # yes
        save_top_dir_dict[94] = "/scr1/Concept2Robot/94/without_force/"#"/juno/group/linshao/ConceptLearning/94" # yes
        save_top_dir_dict[101] = "/juno/group/linshao/ConceptLearning/101" #Yes
        save_top_dir_dict[106] = "/juno/group/linshao/ConceptLearning/106"
        save_top_dir_dict[118] = "/juno/group/linshao/ConceptLearning/118"
        save_top_dir_dict[148] = "/juno/group/linshao/ConceptLearning/148"
        save_top_dir_dict[171] = "/juno/group/linshao/ConceptLearning/171" #Yes
        save_top_dir_dict[172] = "/juno/group/linshao/ConceptLearning/172"

        if self.params.task_id > 0:
            task_id_list = [self.params.task_id]
        else:
            task_id_list =[5, 15, 41, 42, 43, 44, 45, 46, 47, 93, 94, 101, 171]#[5,13,40,42,43,44,45,46,47,86,87,93,94,101,106,118] #[94, 93, 5] #
        len_example_list = {5:1300, \
                            13:1300, \
                            14:2000,\
                            15:2000,\
                            45:2000,\
                            12:1000,\
                            16:1200,\
                            46:1000,\
                            40:2000,\
                            42:2000,\
                            93:2000,\
                            94:2000,\
                            96:1700,\
                            20:1200,\
                            53:2000,\
                            27:900,\
                            86:2000,\
                            87:2000,\
                            88:2000, \
                            89: 2000, \
                            104:2000,\
                            171:2000,\
                            47:1650,\
                            101:2000,\
                            118:1150,\
                            44:1200,\
                            106:1800,\
                            27:1000,\
                            41:1300,\
                            43:2000,\
                            148:2000}
        perf_dict = {5:0.46,
                 13:0.87,
                 14:0.8,
                 15:0.46,
                 20:0.68,#20
                 40:0.5,#40
                 41:0.5, #41
                 42:0.7,#42
                 43:0.45,#43
                 44:0.3,#44
                 45:0.86,#45
                 46:0.84,#46
                 47:1.0,#47,
                 86:0.74,#86,
                 87:1.0,#87
                 88:0.8,
                 89:0.8,
                 93:0.72,#93
                 94:1.0,#94
                 101:1.0,#101
                 106:0.38,#106
                 118:0.39,
                 148:0.8,
                 171:0.8,
                 172:0.8}  # 118  #[1.0, 0.75, 0.55] #

        ###
        perf_list = []
        for task_id in task_id_list:
            perf_list.append(perf_dict[task_id])
        print("task_id_list",task_id_list)
        time.sleep(20)
        task_perf_upbound = np.array(perf_list)
        task_perf_cur = task_perf_upbound * 0.1
        progress = task_perf_cur / task_perf_upbound

        np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')

        example_name_dict = {}
        example_num_dict = {}

        for b_task in task_id_list:
            #save_sub_dir = os.path.join(save_top_dir, str(b_task))
            print(b_task)
            save_sub_dir = save_top_dir_dict[b_task]
            example_name_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) < len_example_list[b_task]]
            example_num_dict[str(b_task)] = len(example_name_dict[str(b_task)])
        print("example_num_dict", example_num_dict)

        for i_iter in range(0, self.params.max_iteration):
            progress = task_perf_cur / task_perf_upbound
            progress = np.minimum(progress, 0.9 * np.ones_like(progress))
            prob = 1.05 - progress
            prob = prob / np.sum(prob)
            task_index_bs = np.random.choice(task_id_list, self.params.batch_size, p=prob)

            task_example_list =  []
            for task_index in task_index_bs:
                task_index_bs1 = np.random.randint(1,example_num_dict[str(task_index)])

                file_path = os.path.join(save_top_dir_dict[task_index],
                                         str(example_name_dict[str(task_index)][task_index_bs1]), "example.npy")
                task_example = np.load(file_path)
                task_example_list.append(task_example)

            imitate_memory = np.array(task_example_list)
            self.agent.imitate_learn(imitate_memory)


            if i_iter % self.params.saving_model_freq_imitation == 0:
                self.agent.save_model_master(i_iter)
                for idx, b_task in enumerate(task_id_list):
                    self.TaskId = b_task
                    perf = self.test()
                    task_perf_cur[idx] = perf

                np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
                np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
                progress = task_perf_cur / task_perf_upbound
                np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')


    def imitate(self):
        imitation_meta_info = "imitation_meta_info.txt"
        save_top_dir_dict = {}
        save_top_dir_dict[5] = "/juno/group/linshao/ConceptLearning/5" #"/scr1/Concept2Robot/5/with_force/rl_2021-01-04_18-11-50"
        save_top_dir_dict[13] = "/juno/group/linshao/ConceptLearning/13"
        save_top_dir_dict[14] = "/juno/group/linshao/ConceptLearning/14"
        save_top_dir_dict[15] = "/juno/group/linshao/ConceptLearning/15" #Yes
        save_top_dir_dict[20] = "/juno/group/linshao/ConceptLearning/20"
        save_top_dir_dict[40] = "/juno/group/linshao/ConceptLearning/40" # Yes
        save_top_dir_dict[41] = "/juno/group/linshao/ConceptLearning/41" # Yes
        save_top_dir_dict[42] = "/juno/group/linshao/ConceptLearning/42" # Yes
        save_top_dir_dict[43] = "/juno/group/linshao/ConceptLearning/43" # Yes
        save_top_dir_dict[44] = "/juno/group/linshao/ConceptLearning/44" # Yes but need to check the gripper
        save_top_dir_dict[45] = "/juno/group/linshao/ConceptLearning/45" # Yes
        save_top_dir_dict[46] = "/juno/group/linshao/ConceptLearning/46" # Yes
        save_top_dir_dict[47] = "/juno/group/linshao/ConceptLearning/47"
        save_top_dir_dict[86] = "/juno/group/linshao/ConceptLearning/86" # No
        save_top_dir_dict[87] = "/juno/group/linshao/ConceptLearning/87" # No
        save_top_dir_dict[88] = "/juno/group/linshao/ConceptLearning/88" # No
        save_top_dir_dict[89] = "/juno/group/linshao/ConceptLearning/89" # No
        save_top_dir_dict[93] = "/juno/group/linshao/ConceptLearning/93" # yes
        save_top_dir_dict[94] = "/scr1/Concept2Robot/94/without_force/"#"/juno/group/linshao/ConceptLearning/94" # yes
        save_top_dir_dict[101] = "/juno/group/linshao/ConceptLearning/101" #Yes
        save_top_dir_dict[106] = "/juno/group/linshao/ConceptLearning/106"
        save_top_dir_dict[118] = "/juno/group/linshao/ConceptLearning/118"
        save_top_dir_dict[148] = "/juno/group/linshao/ConceptLearning/148"
        save_top_dir_dict[171] = "/juno/group/linshao/ConceptLearning/171" #Yes
        save_top_dir_dict[172] = "/juno/group/linshao/ConceptLearning/172"

        if self.params.task_id > 0:
            task_id_list = [self.params.task_id]
        else:
            task_id_list =[5, 15, 41, 42, 43, 44, 45, 46, 47, 93, 94, 101, 171]#[5,13,40,42,43,44,45,46,47,86,87,93,94,101,106,118] #[94, 93, 5] #
        len_example_list = {5:1300, \
                            13:1300, \
                            14:2000,\
                            15:2000,\
                            45:2000,\
                            12:1000,\
                            16:1200,\
                            46:1000,\
                            40:2000,\
                            42:2000,\
                            93:2000,\
                            94:2000,\
                            96:1700,\
                            20:1200,\
                            53:2000,\
                            27:900,\
                            86:2000,\
                            87:2000,\
                            88:2000, \
                            89: 2000, \
                            104:2000,\
                            171:2000,\
                            47:1650,\
                            101:2000,\
                            118:1150,\
                            44:1200,\
                            106:1800,\
                            27:1000,\
                            41:1300,\
                            43:2000,\
                            148:2000}
        perf_dict = {5:0.46,
                 13:0.87,
                 14:0.8,
                 15:0.46,
                 20:0.68,#20
                 40:0.5,#40
                 41:0.5, #41
                 42:0.7,#42
                 43:0.45,#43
                 44:0.3,#44
                 45:0.86,#45
                 46:0.84,#46
                 47:1.0,#47,
                 86:0.74,#86,
                 87:1.0,#87
                 88:0.8,
                 89:0.8,
                 93:0.72,#93
                 94:1.0,#94
                 101:1.0,#101
                 106:0.38,#106
                 118:0.39,
                 148:0.8,
                 171:0.8,
                 172:0.8}  # 118  #[1.0, 0.75, 0.55] #

        ###
        perf_list = []
        for task_id in task_id_list:
            perf_list.append(perf_dict[task_id])
        print("task_id_list",task_id_list)
        time.sleep(20)
        task_perf_upbound = np.array(perf_list)
        task_perf_cur = task_perf_upbound * 0.1
        progress = task_perf_cur / task_perf_upbound

        np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')

        example_name_dict = {}
        example_num_dict = {}

        for b_task in task_id_list:
            #save_sub_dir = os.path.join(save_top_dir, str(b_task))
            print(b_task)
            save_sub_dir = save_top_dir_dict[b_task]
            example_name_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) < len_example_list[b_task]]
            example_num_dict[str(b_task)] = len(example_name_dict[str(b_task)])
        print("example_num_dict", example_num_dict)

        for i_iter in range(0, self.params.max_iteration):
            progress = task_perf_cur / task_perf_upbound
            progress = np.minimum(progress, 0.9 * np.ones_like(progress))
            prob = 1.05 - progress
            prob = prob / np.sum(prob)
            task_index_bs = np.random.choice(task_id_list, self.params.batch_size, p=prob)

            task_example_list =  []
            for task_index in task_index_bs:
                task_index_bs1 = np.random.randint(1,example_num_dict[str(task_index)])

                file_path = os.path.join(save_top_dir_dict[task_index],
                                         str(example_name_dict[str(task_index)][task_index_bs1]), "example.npy")
                task_example = np.load(file_path)
                task_example_list.append(task_example)

            imitate_memory = np.array(task_example_list)
            self.agent.imitate_learn(imitate_memory)


            if i_iter % self.params.saving_model_freq_imitation == 0:
                self.agent.save_model_master(i_iter)
                for idx, b_task in enumerate(task_id_list):
                    self.TaskId = b_task
                    perf = self.test()
                    task_perf_cur[idx] = perf

                np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
                np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
                progress = task_perf_cur / task_perf_upbound
                np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')


    def imitate_feedback(self):
        imitation_meta_info = "imitation_meta_info.txt"
        save_top_dir_dict = {}
        save_top_dir_dict[5] = "/juno/group/linshao/ConceptLearning/5" #"/scr1/Concept2Robot/5/with_force/rl_2021-01-04_18-11-50"
        save_top_dir_dict[13] = "/juno/group/linshao/ConceptLearning/13"
        save_top_dir_dict[14] = "/juno/group/linshao/ConceptLearning/14"
        save_top_dir_dict[15] = "/juno/group/linshao/ConceptLearning/15" #Yes
        save_top_dir_dict[20] = "/juno/group/linshao/ConceptLearning/20"
        save_top_dir_dict[40] = "/juno/group/linshao/ConceptLearning/40" # Yes
        save_top_dir_dict[41] = "/juno/group/linshao/ConceptLearning/41" # Yes
        save_top_dir_dict[42] = "/juno/group/linshao/ConceptLearning/42" # Yes
        save_top_dir_dict[43] = "/juno/group/linshao/ConceptLearning/43" # Yes
        save_top_dir_dict[44] = "/juno/group/linshao/ConceptLearning/44" # Yes but need to check the gripper
        save_top_dir_dict[45] = "/juno/group/linshao/ConceptLearning/45" # Yes
        save_top_dir_dict[46] = "/juno/group/linshao/ConceptLearning/46" # Yes
        save_top_dir_dict[47] = "/juno/group/linshao/ConceptLearning/47"
        save_top_dir_dict[86] = "/juno/group/linshao/ConceptLearning/86" # No
        save_top_dir_dict[87] = "/juno/group/linshao/ConceptLearning/87" # No
        save_top_dir_dict[88] = "/juno/group/linshao/ConceptLearning/88" # No
        save_top_dir_dict[89] = "/juno/group/linshao/ConceptLearning/89" # No
        save_top_dir_dict[93] = "/juno/group/linshao/ConceptLearning/93" # yes
        save_top_dir_dict[94] = "/scr1/Concept2Robot/94/without_force/"#"/juno/group/linshao/ConceptLearning/94" # yes
        save_top_dir_dict[101] = "/juno/group/linshao/ConceptLearning/101" #Yes
        save_top_dir_dict[106] = "/juno/group/linshao/ConceptLearning/106"
        save_top_dir_dict[118] = "/juno/group/linshao/ConceptLearning/118"
        save_top_dir_dict[148] = "/juno/group/linshao/ConceptLearning/148"
        save_top_dir_dict[171] = "/juno/group/linshao/ConceptLearning/171" #Yes
        save_top_dir_dict[172] = "/juno/group/linshao/ConceptLearning/172"

        if self.params.task_id > 0:
            task_id_list = [self.params.task_id]
        else:
            task_id_list =[5, 15, 41, 42, 43, 44, 45, 46, 47, 93, 94, 101, 171]#[5,13,40,42,43,44,45,46,47,86,87,93,94,101,106,118] #[94, 93, 5] #
        len_example_list = {5:1300, \
                            13:1300, \
                            14:2000,\
                            15:2000,\
                            45:2000,\
                            12:1000,\
                            16:1200,\
                            46:1000,\
                            40:2000,\
                            42:2000,\
                            93:2000,\
                            94:2000,\
                            96:1700,\
                            20:1200,\
                            53:2000,\
                            27:900,\
                            86:2000,\
                            87:2000,\
                            88:2000, \
                            89: 2000, \
                            104:2000,\
                            171:2000,\
                            47:1650,\
                            101:2000,\
                            118:1150,\
                            44:1200,\
                            106:1800,\
                            27:1000,\
                            41:1300,\
                            43:2000,\
                            148:2000}
        perf_dict = {5:0.46,
                 13:0.87,
                 14:0.8,
                 15:0.46,
                 20:0.68,#20
                 40:0.5,#40
                 41:0.5, #41
                 42:0.7,#42
                 43:0.45,#43
                 44:0.3,#44
                 45:0.86,#45
                 46:0.84,#46
                 47:1.0,#47,
                 86:0.74,#86,
                 87:1.0,#87
                 88:0.8,
                 89:0.8,
                 93:0.72,#93
                 94:1.0,#94
                 101:1.0,#101
                 106:0.38,#106
                 118:0.39,
                 148:0.8,
                 171:0.8,
                 172:0.8}  # 118  #[1.0, 0.75, 0.55] #

        ###
        perf_list = []
        for task_id in task_id_list:
            perf_list.append(perf_dict[task_id])
        print("task_id_list",task_id_list)
        time.sleep(20)
        task_perf_upbound = np.array(perf_list)
        task_perf_cur = task_perf_upbound * 0.1
        progress = task_perf_cur / task_perf_upbound

        np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
        np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')

        example_name_dict = {}
        example_num_dict = {}

        for b_task in task_id_list:
            #save_sub_dir = os.path.join(save_top_dir, str(b_task))
            print(b_task)
            save_sub_dir = save_top_dir_dict[b_task]
            save_sub_dir_list = [line for line in os.listdir(save_sub_dir) if int(line) < len_example_list[b_task]]
            print("save_sub_dir_list",save_sub_dir_list)
            for sub_dir in save_sub_dir_list:
                save_sub_sub_dir = os.path.join(save_sub_dir, sub_dir)
                example_list = [line for line in os.listdir(save_sub_sub_dir) if line.endswith('feedback.npy')]
                example_name_dict[str(b_task)] = []
                for example_name in example_list:
                    example_name_dict[str(b_task)].append(os.path.join(save_sub_sub_dir, example_name))
                    print("task_path",os.path.join(save_sub_sub_dir, example_name))
            example_num_dict[str(b_task)] = len(example_name_dict[str(b_task)])
        print("example_num_dict", example_num_dict)

        for i_iter in range(0, self.params.max_iteration_feedback):
            progress = task_perf_cur / task_perf_upbound
            progress = np.minimum(progress, 0.9 * np.ones_like(progress))
            prob = 1.05 - progress
            prob = prob / np.sum(prob)
            task_index_bs = np.random.choice(task_id_list, self.params.batch_size, p=prob)

            task_example_list =  []
            for task_index in task_index_bs:
                task_index_bs1 = np.random.randint(1,example_num_dict[str(task_index)])
                file_path = example_name_dict[str(task_index)][task_index_bs1]
                print("file_path",file_path)
                task_example = np.load(file_path)
                task_example_list.append(task_example)

            imitate_memory = np.array(task_example_list)
            self.agent.imitate_learn_feedback(imitate_memory)


            if i_iter % self.params.saving_model_freq_imitation == 0:
                self.agent.save_model_master_feedback(i_iter)
                if 0:
                  for idx, b_task in enumerate(task_id_list):
                    self.TaskId = b_task
                    perf = self.test()
                    task_perf_cur[idx] = perf

                  np.savetxt(os.path.join(self.log_dir, 'successRate_cur.txt'), task_perf_cur, fmt='%1.4f')
                  np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
                  progress = task_perf_cur / task_perf_upbound
                  np.savetxt(os.path.join(self.log_dir, 'progress.txt'), progress, fmt='%1.4f')
