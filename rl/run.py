import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np

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

def load_args():
    parser = argparse.ArgumentParser(description='concept2robot')
    parser.add_argument('--project_dir', default=PROJECT_DIR, type=str, help='project root directory')
    parser.add_argument('--use_cem', action='store_true')

    #### model specification
    parser.add_argument('--max_action', default=0.5, type=float, help='maximum action in translation')
    parser.add_argument('--rotation_max_action', default=0.0, type=float, help='maximum action in rotation')
    parser.add_argument('--a_dim', default=7, type=int, help='action dimension that our model predicts')
    parser.add_argument('--img_w', default=160, type=int)
    parser.add_argument('--img_h', default=120, type=int)
    parser.add_argument('--state_dim', default=120 * 160 * 3, type=int, help='state dimension of the scene image')
    parser.add_argument('--task_dim', default=1024, type=int,
                        help='task description dimension of the language instruction')
    parser.add_argument('--traj_timesteps', default=49, type=int, help='the total timesteps of the trjactory')
    parser.add_argument('--force_term', action='store_true', help='use force term to generate the motion trajectory')
    parser.add_argument('--only_force_term', action='store_true',
                        help='only use force term to generate the motion trajectory')
    parser.add_argument('--feedback_term', action='store_true',
                        help='use feedback term to generate the closed loop motion trajectory')

    ### experiment specification
    parser.add_argument('--classifier', default='video', type=str, choices=['video', 'image'])
    parser.add_argument('--max_ep', default=50000, type=int, help="maximum episode in the training stage")
    parser.add_argument('--max_ep_demon', default=4000, type=int,
                        help='the number of episode generated in the demonstration')
    parser.add_argument('--max_ep_test', default=100, type=int, help='maximum episode in the test stage')

    ## training specification
    parser.add_argument('--a_lr', default=1e-5, type=float, help='the learning rate of the actor')
    parser.add_argument('--c_lr', default=5e-5, type=float, help='the learning rate of the critic')
    parser.add_argument('--traj_lr', default=1e-5, type=float, help='the learning rate of the trajectory')
    parser.add_argument('--explore_var', default=0.5, type=float, help='the exploring variable')
    parser.add_argument('--explore_decay', default=0.9999, type=float, help='the exploring variable')
    parser.add_argument('--start_learning_step', default=2000, type=float, help='start learning step')
    parser.add_argument('--saving_model_freq', default=1000, type=int, help='how often to save the current trained model weight')
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--mem_capacity', default=10000, type=int, help='the capacity of the reply buffer')
    parser.add_argument('--video_reward', action='store_false', help="use video classification as the reward")
    parser.add_argument('--gt_reward', action='store_true',help="use ground truth reward")

    ## testing specification
    parser.add_argument('--restore_step', default=0, type=int, help='restore step')
    parser.add_argument('--restore_path', default="", type=str, help='directory of the pretrained model')


    ### environment or task specification
    parser.add_argument('--recordGif', action='store_true')
    parser.add_argument('--gif_dir', default="../gif_dir", type=str, help="directory of generated gif")
    parser.add_argument('--log_dir', default="../log_dir", type=str)
    parser.add_argument('--save_dir', default="../save_dir", type=str)

    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--task_id', default=5, type=int, help='the task id')
    parser.add_argument('--exp_name', default='without_force', type=str)
    parser.add_argument('--method', default='rl', type=str)
    parser.add_argument('--wid', default=5, type=int, help='wid in pybullet')
    parser.add_argument('--view_point', default='first', type=str, choices=['first', 'third'],
                        help='viewpoint of the camera')
    parser.add_argument('--stage', default='train', type=str, help='which stage to execute')
    args = parser.parse_args()
    return args



class Worker(object):
    def __init__(self, params):
        self.params = params

        if self.params.stage == 'imitation':
            self.file_name = '{}/{}'.format("imitation", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == 'test':
            self.file_name = '{}_{}/{}'.format(str(self.params.task_id), "test", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif self.params.stage == 'train':
            self.file_name = '{}_{}/{}_{}'.format(str(self.params.task_id), str(self.params.exp_name), str(self.params.method),
                                                  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            self.file_name = None

        ### model_dir, writer and log_dir
        if self.params.stage == 'train' or self.params.stage == 'imitation':
            self.writer = SummaryWriter(logdir=os.path.join(self.params.log_dir, self.file_name))
            if self.params.stage == 'train':
               self.model_dir = os.path.join(self.params.save_dir, self.file_name)
               if not os.path.exists(self.model_dir):
                  os.makedirs(self.model_dir)
            else:
                self.model_dir = None
        elif self.params.stage == 'test':
            self.model_dir = None
            self.writer = None
        else:
            self.model_dir = None
            self.writer = None

        self.log_dir = os.path.join(self.params.log_dir, self.file_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.params.writer = self.writer
        self.params.model_dir = self.model_dir

        self.wid = params.wid
        self.TaskId = self.params.task_id
        print("worker id %d" % self.wid, "task id", self.TaskId)
        if self.params.gui:
            self.p_id = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p_id = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.agent = Agent(self.params)
        time_sleep = np.random.uniform(0, 100)
        print("time_sleep", time_sleep)

        env_module = importlib.import_module("env_{}".format(self.TaskId))
        RobotEnv = getattr(env_module, "Engine{}".format(self.TaskId))
        self.env = RobotEnv(worker_id=self.wid, opti=self.params, cReward=self.params.video_reward, p_id=self.p_id,
                            taskId=self.TaskId, n_dmps=7)
        self.task_vec = np.load("../Languages/" + str(self.TaskId) + '.npy')

        self.env_list = {}
        self.p_id_list = {}

    def train(self, restore_step=0, restore_path=None):
        explore_var = self.params.explore_var  # control exploratin
        episode_num = 0
        suc_num = 0
        step_check = 0
        reward_check = 0.0

        if restore_path is not None:
            self.agent.restore(restore_step, restore_path)

        total_step = 0 + restore_step

        if total_step > 0:
            explore_var = explore_var * self.params.explore_decay ** (total_step)

        print("total_step", total_step, "var", explore_var)

        for ep_iter in range(1, self.params.max_ep):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))

            reset_flag = True
            initial_pose = self.env.robotCurrentStatus()
            while True:
                # First generate action
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                action_pred = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim,))
                if total_step < self.params.start_learning_step:
                    action_pred = np.random.uniform(-0.5, 0.5, size=(self.params.traj_timesteps + 1) * self.params.a_dim)
                else:
                    if not self.params.force_term:
                        action_pred[:self.params.a_dim] = self.agent.choose_action(observation, self.task_vec)
                    else:
                        action_pred = self.agent.choose_action(observation, self.task_vec)

                action_clip = action_pred

                # Add exploration
                explore_var = max(explore_var, 0.1)
                action_clip = np.random.normal(action_clip, explore_var)
                self.writer.add_scalar("train/explore_var", explore_var, ep_iter)

                # Clip the action
                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_pred[3:6] * np.pi * 2.0)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action,
                                                               self.params.rotation_max_action)

                action_goal = np.zeros((self.params.a_dim,))
                action_goal = action_clip[:self.params.a_dim]

                action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))

                if self.params.force_term:
                    forces_pred = action_clip[7:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = forces_pred * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)
                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                reset_flag = False
                observation_c = np.copy(observation_next).reshape((-1,))

                ### keep iteration until done
                while not done:
                    coupling_flag = None
                    action_null = np.zeros((self.params.a_dim,))
                    observation_next, reward, done, suc = self.env.step(action_null, None, coupling_flag, reset_flag)
                    if not done:
                        reward = 0

                print("gt_reward",self.params.gt_reward)
                if self.params.gt_reward:
                    reward = float(suc) * 3.0

                print("suc",float(suc),"reward",reward)
                self.writer.add_scalar("train/reward", reward, ep_iter)

                #if not self.params.force_term:
                #    action_penalty = np.sum(np.abs(action[:self.params.a_dim])) / float(
                #        self.params.a_dim) * self.params.action_penalty
                #else:
                #    action_penalty = np.sum(np.abs(action[:self.params.a_dim])) / float(
                #        self.params.a_dim) * self.params.action_penalty + np.sum(
                #        np.abs(action[self.params.a_dim:])) / float(
                #        self.params.a_dim * self.params.traj_timesteps) * self.params.action_penalty * 0.1

                reward = reward#- action_penalty  # -  #np.linalg.norm(self.env.robotCurrentStatus() - self.env.dmp.goal[:3]) * 0.5

                observation_next = np.reshape(observation_next, (-1,))
                #print("taskid", self.TaskId, "exp_name", self.params.exp_name, "ep_", ep_, " action", action[:7],
                #      "action_", action_[:7], "reward", reward, "suc", suc, "dmp dist",
                #      np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]),
                #     self.env.robotCurrentStatus()[:3], self.env.dmp.goal[:3])

                while len(self.env.real_traj_list) < self.env.dmp.timesteps:
                    self.env.real_traj_list.append(self.env.real_traj_list[-1])

                ### traj_real is different from traj generated because
                traj_real = np.array(self.env.real_traj_list)
                traj_real = traj_real.reshape((-1,))

                step_check += 1
                reward_check += reward
                self.agent.store_transition(initial_pose, traj_real, observation, action_pred, reward, done, self.task_vec)

                if ep_iter > self.params.start_learning_step:
                    explore_var = explore_var * self.params.explore_decay  # decay the action randomness
                    a = self.agent.learn()
                    print("starting to learn")

                observation = observation_next
                total_step += 1

                if ep_iter % self.params.saving_model_freq == 0:
                    current_performance = float(suc_num) / (episode_num + 0.0001)
                    print("suc rate %f reward %f var %f" % (current_performance, float(reward_check) / (episode_num + 0.0001), explore_var))
                    print("saving models at step%d" % ep_iter)
                    self.agent.save_model(ep_iter)
                    self.test(ep_iter)

                if done:
                    self.writer.add_scalar("train/success_rate", suc, ep_iter)
                    suc_num += suc
                    episode_num += 1
                    break

    def test(self, eval_step=0, restore_path=None):
        total_suc = 0

        if restore_path is not None:
            self.agent.restore(eval_step, restore_path)
            print("testing from restoring sth", restore_path)

        for ep_ in range(self.params.max_ep_test):
            observation = self.env.reset()
            observation = np.reshape(observation, (-1,))
            reset_flag = True

            tt = 0
            while True:
                action_clip = np.zeros(((self.params.traj_timesteps + 1) * self.params.a_dim, ))

                # Added exploration noise
                action_pred = self.agent.choose_action(observation, self.task_vec)

                ### clip by the max action
                if self.params.force_term:
                    action_clip = action_pred
                else:
                    action_clip[:self.params.a_dim] = action_pred

                action_clip[:3] = action_clip[:3].clip(-self.params.max_action, self.params.max_action)
                action_clip[3:6] = (action_clip[3:6] * np.pi * 2.0).clip(-self.params.rotation_max_action, self.params.rotation_max_action)

                action_goal = np.zeros((7,))
                action_goal[:7] = action_clip[:7]

                action_forces = np.zeros((self.params.traj_timesteps, self.params.a_dim))

                if self.params.force_term:
                    f_a_3 = action_clip[self.params.a_dim:].reshape((self.params.a_dim, self.params.traj_timesteps)).transpose() * 50.0
                    weights = np.linspace(1, 0, self.params.traj_timesteps).reshape((self.params.traj_timesteps, 1))
                    action_forces[:, :7] = f_a_3 * weights
                    action_forces[:, 3:6] = (action_forces[:, 3:6]).clip(-self.params.rotation_max_action / np.pi * 50.0,
                                                     self.params.rotation_max_action / np.pi * 50.0)
                    observation_next, reward, done, suc = self.env.step(action_goal, action_forces, None, reset_flag)
                else:
                    observation_next, reward, done, suc = self.env.step(action_goal, None, None, reset_flag)

                reset_flag = False
                observation_c = np.copy(observation_next).reshape((-1,))

                while not done:
                    coupling_ = np.zeros((7,))
                    observation_next, reward, done, suc = self.env.step(np.zeros((7,)), None, coupling_, reset_flag)

                print("Tesing", "taskid", self.TaskId, "ep_", ep_, " action", action_clip[:7], "action_pred", action_pred[:7],
                      "reward", reward, "suc", suc, "dmp dist",
                      np.linalg.norm(self.env.robotCurrentStatus()[:3] - self.env.dmp.goal[:3]),
                      self.env.robotCurrentStatus()[:3], self.env.dmp.goal[:3])

                observation = observation_next

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

        perf = total_suc / float(self.params.max_ep_test)
        print("success performance", perf)
        np.savetxt(os.path.join(self.log_dir,'successRate.txt'), np.array([perf]), fmt='%1.4f')
        print("successfully done!!!!!")


if __name__ == "__main__":
    args = load_args()
    print(args.use_cem)
    print(args.max_ep)
    #args.gui = True
    #args.restore_step = 39000
    #args.restore_path = '/juno/u/lins2/toki/ConceptManipulation/rl/asc/save_model/5_VR/ASC_2020-06-13_10-44-05'
    #args.stage = 'test'
    worker = Worker(args)
    if args.stage == 'test':
        worker.test(args.restore_step, args.restore_path)
    elif args.stage == 'train':
        worker.train()
    else:
        pass