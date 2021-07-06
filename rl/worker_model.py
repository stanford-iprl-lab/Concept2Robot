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
from ddmp import DDMP as DMP

sim_DIR = os.path.join(BASE_DIR, "../external/bullet3_default/")
sys.path.insert(0,os.path.join(sim_DIR,'bullet3/build_cmake/examples/pybullet'))
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

def mse_metric(actual, predicted):
    sum_error = 0.0
    # loop over all values
    for i in range(len(actual)):
        # the error is the sum of (actual - prediction)^2
        prediction_error =  actual[i] - predicted[i]
        sum_error += (prediction_error ** 2)
    # now normalize
    mean_error = sum_error / float(len(actual))
    return (mean_error)

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
        elif self.params.stage == 'test' or self.params.stage == "imitation_test":
            self.file_name = '{}_{}/{}'.format(str(self.params.task_id), "test", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == 'train':
            print("self.params.comments",self.params.comment)
            self.file_name = '{}_{}/{}_{}_{}'.format(str(self.params.task_id), str(self.params.exp_name), str(self.params.method), self.params.comment,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
        elif self.params.stage == 'test' or self.params.stage == 'feedback_test' or self.params.stage == 'imitation_test_old':
            self.model_dir = None
            self.writer = None
        elif self.params.stage == "imitation_test":
            self.writer = SummaryWriter(logdir=os.path.join(self.params.log_dir, self.file_name))
            self.model_dir = os.path.join(self.params.save_dir, self.file_name)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
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
        self.agent_expert = Agent(self.params)
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


    def train(self, restore_episode=0, restore_path=None, restore_episode_goal=0, restore_path_goal=None):
      if self.params.force_term:
        self.train_force(restore_episode=restore_episode, restore_path=restore_path, restore_episode_goal=restore_episode_goal, restore_path_goal=restore_path_goal)
      else:
        explore_var = self.params.explore_var  # control exploratiion
        episode_num = 0
        suc_num = 0
        step_check = 0
        reward_check = 0

        if restore_path is not None:
            self.agent.restore(restore_episode, restore_path)

        total_episode = 0 + restore_episode

        #if total_episode > 0:
            #explore_var = explore_var * self.params.explore_decay ** (total_episode)

        print("total_episode", total_episode, "var", explore_var)

        for ep_iter in range(1, self.params.max_ep):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()

            while True:
                # First generate action
                if total_episode < self.params.start_learning_episode:
                    goal_pred = np.random.uniform(-0.5, 0.5, size=self.params.a_dim)
                    #force_pred = np.random.uniform(-1.0, 1.0, size=(self.params.traj_timesteps, self.params.a_dim))
                    #weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(self.params.traj_timesteps)
                    #force_pred = force_pred * weights
                else:
                    explore_var = max(explore_var, 0.2)
                    goal_pred, _ = self.agent.choose_action(observation, self.task_vec)
                    goal_pred = np.random.normal(goal_pred, explore_var)

                force_pred = np.zeros((self.params.traj_timesteps, self.params.a_dim))

                # Clip by max action
                goal_pred[:3] = goal_pred[:3].clip(-self.params.max_action, self.params.max_action)
                goal_pred[3:6] = goal_pred[3:6].clip(-self.params.rotation_max_action, self.params.rotation_max_action)

                self.writer.add_scalar("train/explore_var", explore_var, ep_iter)

                observation_next, reward, done, suc = self.env.step(goal_pred, None, None, reset_flag)

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
                action_penalty = np.mean(np.abs(goal_pred)) * self.params.action_penalty
                action_penalty2 = np.mean(np.abs(force_pred)) / float(self.params.traj_timesteps) * 0.3 * self.params.action_penalty

                observation_next = np.reshape(observation_next, (-1,))

                # traj_real is different from traj generated
                while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                    self.env.real_traj_list.append(self.env.real_traj_list[-1])
                traj_real = np.array(self.env.real_traj_list)

                traj_pred_dmp = DMP(self.params, n_dmps=self.params.a_dim, goal=initial_pose + goal_pred,
                                    start=initial_pose, force=None, timesteps=self.params.traj_timesteps)
                traj_pred = traj_pred_dmp.rollout()[0]
                traj_diff = np.mean((traj_pred - traj_real) ** 2) * float(suc)
                traj_diff_penalty = traj_diff
                reward = reward - action_penalty# - action_penalty2

                print("Training", "success", suc, "taskid", self.TaskId, "ep_iter", ep_iter, "action_pred", goal_pred,
                      "reward", reward, "suc", suc, "action_penalty", action_penalty, "action_penalty2", action_penalty2)
                print(force_pred[0])
                print(force_pred[1])
                print(force_pred[2])
                print(force_pred[20])
                print(force_pred[30])
                print(force_pred[40])
                print(force_pred[47])
                self.writer.add_scalar("train_action_0/pred_action", goal_pred[0], ep_iter)
                self.writer.add_scalar("train_action_1/pred_action", goal_pred[1], ep_iter)
                self.writer.add_scalar("train_action_2/pred_action", goal_pred[2], ep_iter)
                self.writer.add_scalar("train_action_6/pred_action", goal_pred[6], ep_iter)
                self.writer.add_scalar("train_reward/action_penalty", action_penalty, ep_iter)
                self.writer.add_scalar("train_reward/action_penalty2", action_penalty2, ep_iter)
                self.writer.add_scalar("train_reward/traj_diff_penalty", traj_diff_penalty, ep_iter)

                traj_real = traj_real.reshape((-1,))

                step_check += 1
                reward_check += reward

                # Important!! the shape of action_pred is ((self.params.traj_timesteps + 1) * self.params.a_dim)
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred[:self.params.a_dim] = goal_pred
                action_pred[self.params.a_dim:] = force_pred.reshape((-1,))

                self.agent.store_transition(initial_pose, traj_real, observation, action_pred, reward, done, self.task_vec, goal_pred)

                if ep_iter > self.params.start_learning_episode:
                    #explore_var = explore_var * self.params.explore_decay  # decay the action randomness
                    a = self.agent.learn()
                    print("starting to learn")

                observation = observation_next
                total_episode += 1

                if ep_iter % self.params.saving_model_freq == 0:
                    current_performance = float(suc_num) / (episode_num + 0.0001)
                    print("suc rate %f reward %f var %f" % (current_performance, float(reward_check) / (episode_num + 0.0001), explore_var))
                    print("saving models at step %d" % ep_iter)
                    self.agent.save_model_actor_critic(ep_iter)
                    print("restore_episode",ep_iter)
                    cur_perf = self.test(restore_episode=ep_iter)
                    self.writer.add_scalar("test/success_rate", cur_perf, ep_iter)

                if done:
                    self.writer.add_scalar("train/success_rate", suc, ep_iter)
                    suc_num += suc
                    episode_num += 1
                    break

    def train_force(self, restore_episode=0, restore_path=None, restore_episode_goal=0, restore_path_goal=None):
        print("restiore_episode_goal",restore_episode_goal)
        self.agent.restore_actor_goal_only(step=restore_episode_goal, restore_path=restore_path_goal)
        self.agent.restore_actor(step=restore_episode_goal, restore_path=restore_path_goal)
        #self.agent.memory = np.load('memeory.npy')
        #print(self.agent.memory[0])
        #print(self.agent.memory[1])
        self.agent.pointer += 990

        explore_var = self.params.explore_var  # control exploratin
        episode_num = 0
        suc_num = 0
        step_check = 0
        reward_check = 0

        if restore_path is not None and restore_episode > 0:
            self.agent.restore_force(restore_episode, restore_path)

        total_episode = 0 #+ restore_episode

        #if total_episode > 0:
        #    explore_var = explore_var * self.params.explore_decay ** (total_episode)

        print("total_episode", total_episode, "var", explore_var)

        for ep_iter in range(1, self.params.max_ep):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()

            while True:
                # First generate action
                if total_episode < self.params.start_learning_episode:
                    goal_pred, _ = self.agent.choose_action(observation, self.task_vec)
                    goal_pred_goal_only = goal_pred
                    force_pred = np.random.uniform(-1.0, 1.0, size=(self.params.traj_timesteps, self.params.a_dim)) 
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(self.params.traj_timesteps)
                    force_pred = force_pred * weights
                else:
                    explore_var = max(explore_var, 0.1)
                    goal_pred_goal_only = self.agent.choose_action_goal_only(observation, self.task_vec)
                    goal_pred, force_pred = self.agent.choose_action(observation, self.task_vec)
                    scales = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(self.params.traj_timesteps) * explore_var
                    scales = np.tile(scales, [1,self.params.a_dim])
                    action_max = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1)) * float(self.params.traj_timesteps)
                    action_max = np.tile(action_max, [1,self.params.a_dim])
                    force_pred = np.random.normal(force_pred, scales)
                    force_pred = force_pred.clip(-action_max, action_max)

                print("goal_pred_goal_only",goal_pred_goal_only)
                self.writer.add_scalar("train/explore_var", explore_var, ep_iter)

                observation_next, reward, done, suc = self.env.step(goal_pred, force_pred, None, reset_flag)

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
                    action_penalty = np.sum(np.abs(goal_pred[:self.params.a_dim])) / float(
                        self.params.a_dim) * self.params.action_penalty

                observation_next = np.reshape(observation_next, (-1,))

                # traj_real is different from traj generated
                while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                    self.env.real_traj_list.append(self.env.real_traj_list[-1])
                traj_real = np.array(self.env.real_traj_list)

                traj_pred_dmp = DMP(self.params, n_dmps=self.params.a_dim, goal=initial_pose + goal_pred,
                                    start=initial_pose, force=None, timesteps=self.params.traj_timesteps)
                traj_pred = traj_pred_dmp.rollout()[0]
                traj_diff = np.mean((traj_pred - traj_real) ** 2) * float(suc)
                #if traj_diff > 0.1:
                #    traj_diff = 0.1
                traj_diff_penalty = traj_diff #* self.params.action_penalty

                action_penalty = np.mean(np.abs(force_pred)) / float(self.params.traj_timesteps) * 0.1

                reward = reward - action_penalty #- traj_diff_penalty
                print(force_pred[0])
                print(force_pred[1])
                print(force_pred[2])
                print(force_pred[20])
                print(force_pred[30])
                print(force_pred[40])
                print(force_pred[47])
                print("Training", "success", suc, "taskid", self.TaskId, "ep_iter", ep_iter, "action_pred", goal_pred,
                      "reward", reward, "suc", suc, "action_penalty", action_penalty, "traj_diff_penalty", traj_diff_penalty)

                self.writer.add_scalar("train_action_0/pred_action", goal_pred[0], ep_iter)
                self.writer.add_scalar("train_action_1/pred_action", goal_pred[1], ep_iter)
                self.writer.add_scalar("train_action_2/pred_action", goal_pred[2], ep_iter)
                self.writer.add_scalar("train_action_6/pred_action", goal_pred[6], ep_iter)
                self.writer.add_scalar("train_action_0/gt_action", goal_pred_goal_only[0], ep_iter)
                self.writer.add_scalar("train_action_1/gt_action", goal_pred_goal_only[1], ep_iter)
                self.writer.add_scalar("train_action_2/gt_action", goal_pred_goal_only[2], ep_iter)
                self.writer.add_scalar("train_action_6/gt_action", goal_pred_goal_only[6], ep_iter)
                self.writer.add_scalar("train_action_0/pred_force", np.mean(force_pred[0]), ep_iter)
                self.writer.add_scalar("train_action_1/pred_force", np.mean(force_pred[1]), ep_iter)
                self.writer.add_scalar("train_action_2/pred_force", np.mean(force_pred[2]), ep_iter)
                self.writer.add_scalar("train_action_6/pred_force", np.mean(force_pred[6]), ep_iter)
                self.writer.add_scalar("train_reward/action_penalty", action_penalty, ep_iter)
                self.writer.add_scalar("train_reward/traj_diff_penalty", traj_diff_penalty, ep_iter)

                traj_real = traj_real.reshape((-1,))

                step_check += 1
                reward_check += reward

                # Important!! the shape of action_pred is ((self.params.traj_timesteps + 1) * self.params.a_dim)
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred[:self.params.a_dim] = goal_pred
                action_pred[self.params.a_dim:] = force_pred.reshape((-1,))

                action_gt = np.zeros((self.params.a_dim,))
                action_gt = goal_pred_goal_only
                self.agent.store_transition(initial_pose, traj_real, observation, action_pred, reward, done, self.task_vec, action_gt)

                if ep_iter > self.params.start_learning_episode:
                    #explore_var = explore_var * self.params.explore_decay  # decay the action randomness
                    a = self.agent.learn()
                    print("starting to learn")

                observation = observation_next
                total_episode += 1

                if ep_iter % self.params.saving_model_freq == 0:
                    #np.save("memeory.npy",self.agent.memory)
                    current_performance = float(suc_num) / (episode_num + 0.0001)
                    print("suc rate %f reward %f var %f" % (current_performance, float(reward_check) / (episode_num + 0.0001), explore_var))
                    print("saving models at step%d" % ep_iter)
                    self.agent.save_model_actor_critic(ep_iter)
                    cur_perf = self.test(ep_iter, restore_episode_goal=restore_episode_goal, restore_path_goal=restore_path_goal)
                    self.writer.add_scalar("test/success_rate", cur_perf, ep_iter)

                if done:
                    self.writer.add_scalar("train/success_rate", suc, ep_iter)
                    suc_num += suc
                    episode_num += 1
                    break

    def test(self, restore_episode=0, restore_path=None, restore_episode_goal=0, restore_path_goal=None):
        total_suc = 0

        if self.params.stage == "imitation" or self.params.stage == "imitation_test":
            self.p_id.disconnect()
            self.p_id.__del__()
            del self.env
            print("self.TaskId",self.TaskId)
            self.env = Enver(self.TaskId, self.params).env

        if restore_path is not None:
            if self.params.stage == "imitation_test":
                print("restore_episode",restore_episode)
                self.agent.restore_master(restore_episode, restore_path)
                self.agent.master.eval()
            else:
                print("restore_episode_goal", restore_episode_goal)
                print("restore_path_goal", restore_path_goal)
                if restore_path is None:
                    self.agent.restore_actor(restore_episode)
                    print("testing from restoring ", self.params.model_dir)
                else:
                    print("restore_episode",restore_episode)
                    print("restore_path", restore_path)
                    self.agent.restore_actor(restore_episode, restore_path)
                    print("testing from restoring sth", restore_path)

        if self.params.stage == "demonstration":
            max_iteration = self.params.max_ep_demon
        elif self.params.stage == "test" or self.params.stage == "train":
            max_iteration = self.params.max_ep_test
        elif self.params.stage == "imitation" or self.params.stage == "imitation_test":
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
                if self.params.stage == "imitation" or self.params.stage == "imitation_test":
                    goal_pred, force_pred = self.agent.choose_action_master(observation, self.task_vec)
                else:
                    if self.params.force_term:
                        goal_pred, force_pred = self.agent.choose_action(observation, self.task_vec)
                    else:
                        goal_pred, _ = self.agent.choose_action(observation, self.task_vec)
                        force_pred = np.zeros((self.params.traj_timesteps, self.params.a_dim))

                if self.params.force_term or self.params.stage == "imitation" or self.params.stage == "imitation_test":
                    observation_next, reward, done, suc = self.env.step(goal_pred, force_pred, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(goal_pred, None, None, reset_flag)

                reset_flag = False
                observation_c = np.copy(observation_next).reshape((-1,))

                while not done:
                    feedback_term = np.zeros((7,))
                    observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, feedback_term, reset_flag)
                print(force_pred[0])
                print(force_pred[1])
                print(force_pred[2])
                print(force_pred[20])
                print(force_pred[30])
                print(force_pred[40])
                print(force_pred[47])
                print("Tesing", "success", suc, "taskid", self.TaskId, "ep_iter", ep_iter, " goal_pred", goal_pred, "force_pred", force_pred[0],
                      "reward", reward, "suc", suc)

                observation = observation_next

                if suc and self.params.stage == "demonstration":
                    # generate trajectory
                    while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                        self.env.real_traj_list.append(self.env.real_traj_list[-1])
                    traj_real = np.array(self.env.real_traj_list)
                    traj_real = traj_real.reshape((-1,))

                    action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                    action_pred[:self.params.a_dim] = goal_pred
                    action_pred[self.params.a_dim:] = force_pred.reshape((-1,))
                    print("action_pred_force",action_pred[self.params.a_dim:])

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
        if (not self.params.stage == 'imitation') or (not self.params.stage == 'imitation_test'):
            np.savetxt(os.path.join(self.log_dir,'successRate.txt'), np.array([perf]), fmt='%1.4f')

        #self.agent.master.train()
        print("successfully done!!!!!")
        return perf

    def imitate(self):
        imitation_meta_info = "imitation_meta_info.txt"
        pretrained_model_list = "pretrained_model_list.txt"
        meta_info = [line.strip().split() for line in open("pretrained_model_list.txt")]
        save_top_dir_dict = {}
        save_model_dir = {}
        len_example_list = {}
        perf_dict = {}
        for meta_i in meta_info:
           print(meta_i)
           meta_i[0] = int(meta_i[0])
           save_top_dir_dict[meta_i[0]] = meta_i[4]
           save_model_dir[meta_i[0]] = (meta_i[3], meta_i[3])
           len_example_list[meta_i[0]] = int(len([line for line in os.listdir(meta_i[4])]) * 0.99)
           perf_dict[meta_i[0]] = float(meta_i[1])
           print(save_top_dir_dict[meta_i[0]])
           print(save_model_dir[meta_i[0]])

        restore_old = {}

        if 0:#self.params.task_id > 0:
            task_id_list = [self.params.task_id]
        else:
            task_id_list = [meta_i[0] for meta_i in meta_info]#[45,43,600,604,608,609,610,611,86]#

        ###
        perf_list = []
        error_list = []
        for task_id in task_id_list:
            perf_list.append(perf_dict[task_id])
            error_list.append(0.1)

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
            print(b_task)
            save_sub_dir = save_top_dir_dict[b_task]
            example_name_dict[str(b_task)] = [line for line in os.listdir(save_sub_dir) if int(line) < len_example_list[b_task]]
            example_num_dict[str(b_task)] = len(example_name_dict[str(b_task)])
        print("example_num_dict", example_num_dict)

        self.agent.master.train()
        for i_iter in range(0, self.params.max_iteration):
            print("iter", i_iter, "max_iteration", self.params.max_iteration)
            progress = task_perf_cur / task_perf_upbound
            progress = np.minimum(progress, 0.9 * np.ones_like(progress))
            prob = np.array(error_list)
            prob = prob / np.sum(prob)
            prob = np.maximum(prob, np.ones_like(prob) * 0.1)
            prob = prob / np.sum(prob)
            bs_dict = {}
            for task_id in task_id_list:
                bs_dict[task_id] = 0
            task_index_bs = np.random.choice(task_id_list, self.params.batch_size, p=prob)
            #task_index_bs = np.random.choice(task_id_list, self.params.batch_size)
            for bi in task_index_bs:
                bs_dict[bi] += (1.0/float(self.params.batch_size))
            print_string = []
            for task_id, progress_id, prob_id, error_id in zip(task_id_list, progress, prob, error_list):
                print_string.append((task_id, progress_id, prob_id, bs_dict[task_id], error_id))
            print(print_string)
            task_example_list =  []
            for task_index in task_index_bs:
                task_index_bs1 = np.random.randint(1,example_num_dict[str(task_index)])

                file_path = os.path.join(save_top_dir_dict[task_index],
                                         str(example_name_dict[str(task_index)][task_index_bs1]), "example.npy")
                task_example = np.load(file_path)
                task_example_list.append(task_example)

            imitate_memory = np.array(task_example_list)
            self.agent.imitate_learn(imitate_memory)


            if i_iter % self.params.saving_model_freq_imitation == self.params.saving_model_freq_imitation - 1:
                self.agent.master.eval()
                eval_error_list = []
                for task_id in task_id_list:
                    print("TTTTTTTTTTTTTTTTTTTTTTTTTT ",task_id)
                    task_index_bs = np.random.randint(1, example_num_dict[str(task_id)],
                                                       size=self.params.batch_size)
                    task_example_list = []
                    for task_index in task_index_bs:
                        file_path = os.path.join(save_top_dir_dict[task_id],
                                             str(example_name_dict[str(task_id)][task_index]), "example.npy")
                        task_example = np.load(file_path)
                        task_example_list.append(task_example)

                    imitate_memory = np.array(task_example_list)
                    eval_goal_loss, eval_force_loss, eval_loss = self.agent.imitate_learn_test(imitate_memory)
                    print("eval_loss",eval_loss)
                    eval_error_list.append(eval_loss)
                self.agent.save_model_master(i_iter)
                print("saving model of imitation")
                for idx, b_task in enumerate(task_id_list):
                    self.TaskId = b_task
                    del self.agent_expert
                    self.agent_expert = Agent(self.params)
#                    if restore_old[self.TaskId] or self.TaskId not in restore_old:
#                        self.params.restore_old = True
                    perf = self.test()#restore_episode=save_model_dir[self.TaskId][1], restore_path=save_model_dir[self.TaskId][0])
                    #error = 0.001
                    #error_list[idx] = error
                    task_perf_cur[idx] = perf
                for (task_id, eval_err, test_err) in zip(task_id_list, eval_error_list, error_list):
                    print("task_id",task_id, "eval_err", eval_err, "test_error", test_err)
                #self.params.writer.add_scalar('test_perf',perf,i_iter)
                np.savetxt(os.path.join(self.log_dir, 'successRate_cur_'+str(i_iter)+'.txt'), task_perf_cur, fmt='%1.4f')
                np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
                progress = task_perf_cur / task_perf_upbound
                np.savetxt(os.path.join(self.log_dir, 'progress_'+str(i_iter)+'.txt'), progress, fmt='%1.4f')
                self.agent.master.eval()

    def imitate_test(self,restore_episode,restore_path=None):
        print("restore_episode", restore_episode)
        self.agent.restore_master(restore_episode, restore_path)
        ###

        pretrained_model_list = "pretrained_model_list.txt"
        meta_info = [line.strip().split() for line in open("pretrained_model_list.txt")]
        save_top_dir_dict = {}
        save_model_dir = {}
        len_example_list = {}
        perf_dict = {}
        for meta_i in meta_info:
           print(meta_i)
           meta_i[0] = int(meta_i[0])
           perf_dict[meta_i[0]] = float(meta_i[1])

        if 0:
            task_id_list = [self.params.task_id]
        else:
            task_id_list = [meta_i[0] for meta_i in meta_info]

        print("task_id_list",task_id_list)
        perf_list = []

        for task_id in task_id_list:
            perf_list.append(perf_dict[task_id])
        task_perf_upbound = np.array(perf_list)
        task_perf_cur = task_perf_upbound * 0.1
        example_name_dict = {}
        example_num_dict = {}
        self.agent.master.eval()


        for i_iter in range(0, self.params.max_iteration):
            if i_iter % self.params.saving_model_freq_imitation == self.params.saving_model_freq_imitation - 1:
                for idx, b_task in enumerate(task_id_list):
                    self.TaskId = b_task
                    del self.agent_expert
                    self.agent_expert = Agent(self.params)
                    perf = self.test()
                    task_perf_cur[idx] = perf

                np.savetxt(os.path.join(self.log_dir, 'successRate_cur_'+str(i_iter)+'.txt'), task_perf_cur, fmt='%1.4f')
                np.savetxt(os.path.join(self.log_dir, 'successRate_upbound.txt'), task_perf_upbound, fmt='%1.4f')
                progress = task_perf_cur / task_perf_upbound
                np.savetxt(os.path.join(self.log_dir, 'progress_'+str(i_iter)+'.txt'), progress, fmt='%1.4f')

