import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sys
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import torch.optim as optim
from torch.optim import Adam

criterion = nn.MSELoss()

os.environ["OMP_NUM_THREADS"] = "1"
device=torch.device("cuda")

np.set_printoptions(precision=4,suppress=True)

def v_wrap(np_array,dtype=np.float32):
  if np_array.dtype != dtype:
    np_array = np_array.astype(dtype)
  return torch.from_numpy(np_array).to(device)


class DDMP():
  def __init__(self,opti,n_dmps,start=None,goal=None,force=None,timesteps=None):
    self.opti = opti
    self.n_dmps = n_dmps
    if timesteps is not None:
        self.timesteps = timesteps
    else:
        self.timesteps = self.opti.dmp_timesteps
    self.dt = 1./ (self.timesteps)
    self.ay = 12.
    self.by = self.ay / 4.
    if opti is not None:
        self.time_param = self.opti.dmp_time_param
    else:
        self.time_param = 't'
    if goal is not None:
      self.goal = goal
    else:
      self.goal = np.ones((n_dmps,))
    if start is not None:
      self.start = start
    else:
      self.start = np.zeros((n_dmps,))
    if force is not None:
      self.force = force
    else:
      self.force = np.zeros((self.timesteps,self.n_dmps))
    self.timestep = 0
    self.grad_y_goal = None
    self.grad_y_force = None


  def gen_goal(self,y_des):
    return np.copy(y_des[:,-1])


  def set_start(self,start=None):
    if start is not None:
      self.start = start


  def set_goal(self,goal=None):
    if goal is not None:
      self.goal = goal


  def set_force(self,force=None):
    if force is not None:
      for i in range(self.n_dmps):
        self.force[:,i] = force[:,i]
      if 0:
        fig = plt.figure(1)
        lenT = self.force.shape[0]
        plt.plot(np.arange(0,lenT),self.force[:,0],color='red')
        plt.plot(np.arange(0,lenT),self.force[:,1],color='green')
        fig.canvas.draw()
        images = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3)
        cv2.imshow("Example",images)
        fig.canvas.figure.clf()
        cv2.waitKey(1)


  def reset_state(self):
    """Reset the system state."""
    self.y = np.copy(self.start)
    self.dy = np.zeros(self.n_dmps)
    self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + self.force[0]
    self.timestep = 0
 
  def rollout(self,timesteps=None):
    self.reset_state()

    timesteps = self.timesteps
    # set up tracking vectors
    y_track = np.zeros((timesteps, self.n_dmps)) 
    dy_track = np.zeros((timesteps, self.n_dmps)) 
    ddy_track = np.zeros((timesteps, self.n_dmps)) 

    # at initila step
    y_track[0] = self.y.copy()
    dy_track[0] = self.dy.copy()
    ddy_track[0] = self.ddy.copy()

    for t in range(1,timesteps):
      y_track[t], dy_track[t], ddy_track[t] = self.step()

    if 0:
        fig = plt.figure(1)
        lenT = y_track.shape[0]
        plt.plot(np.arange(0,lenT),y_track[:,0],color='red')
        plt.plot(np.arange(0,lenT),y_track[:,1],color='green')
        plt.plot(np.arange(0,lenT),y_track[:,2],color='blue')
        fig.canvas.draw()
        images = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3)
        cv2.imshow("Example",images)
        fig.canvas.figure.clf()
        cv2.waitKey(1)

    return y_track, dy_track, ddy_track

 
  def step(self,coupling=None):
    s = self.timestep
    self.timestep += 1
    self.ddy = self.ay * (self.by * (self.goal - self.y) - self.dy) + self.force[s]
    self.dy += self.ddy * self.dt
    self.y += self.dy * self.dt
    if coupling is not None:
        self.y += coupling

    return self.y, self.dy, self.ddy

  def gen_traj(self):
    return self.rollout()[0]

  def gradient_goal(self,GT_traj):
    pred_traj = self.rollout()[0]
    assert GT_traj.shape == pred_traj.shape

    if self.grad_y_goal is None:
        grad_ddy = self.ay * self.by
        grad_dy = grad_ddy * self.dt
        grad_y = np.zeros(self.timesteps)
        grad_y[0] = grad_dy * self.dt

        for t in range(1,self.timesteps):
            grad_ddy = self.ay * (self.by * (1. - grad_y[t-1]) - grad_dy)
            grad_dy += grad_ddy * self.dt
            grad_y[t] = grad_y[t-1] + grad_dy * self.dt

        self.grad_y_goal = np.expand_dims(grad_y, 1)

    grad_goal = np.sum((pred_traj - GT_traj) * self.grad_y_goal, axis=0)
    return grad_goal

  def gradient_goal_v2(self,GT_traj):
    grad_goal = self.goal - GT_traj[-1]
    return grad_goal

  def gradient_force(self, GT_traj):
    pred_traj = self.rollout()[0]
    assert GT_traj.shape == pred_traj.shape

    if self.grad_y_force is None:
        grad_ddy = np.zeros(self.timesteps)
        grad_ddy[0] = 1.
        grad_dy = grad_ddy * self.dt
        grad_y = np.zeros((self.timesteps,self.timesteps))
        grad_y[0] = grad_dy * self.dt

        for t in range(1,self.timesteps):
            grad_ddy = self.ay * (self.by * -grad_y[t-1] - grad_dy)
            grad_ddy[t] += 1.
            grad_dy += grad_ddy * self.dt
            grad_y[t] = grad_y[t-1] + grad_dy * self.dt

        self.grad_y_force = np.expand_dims(grad_y, 2)
    grad_force = np.sum(np.expand_dims(pred_traj - GT_traj, 1) * self.grad_y_force, axis=0)
    return grad_force

  def gradient_force_v2(self, GT_traj):
    pred_traj = self.rollout()[0]
    assert GT_traj.shape == pred_traj.shape

    if self.grad_y_force is None:
        grad_ddy = np.zeros(self.timesteps)
        grad_ddy[0] = 1.
        grad_dy = grad_ddy * self.dt
        grad_y = np.zeros((self.timesteps,self.timesteps))
        grad_y[0] = grad_dy * self.dt

        for t in range(1,self.timesteps):
            grad_ddy = self.ay * (self.by * -grad_y[t-1] - grad_dy)
            grad_ddy[t] += 1.
            grad_dy += grad_ddy * self.dt
            grad_y[t] = grad_y[t-1] + grad_dy * self.dt

        self.grad_y_force = np.expand_dims(grad_y, 2)
    grad_force = np.sum(np.expand_dims(pred_traj - GT_traj, 1) * self.grad_y_force, axis=0)

    grad_force_2 = self.grad_y_force[-1] * np.expand_dims((pred_traj[-1]-self.goal),axis=0)
    grad_force = grad_force + grad_force_2
    return grad_force_2

  def imitate_path(self,y_gt,plot=False):
      goal_des = y_gt[-1].copy()
      y_des = y_gt.copy()  # T
      dy_des = np.diff(y_des, axis=0) / self.dt  # T-1
      dy_des = np.vstack((np.zeros((1, self.n_dmps)), dy_des))  # T
      ddy_des = np.diff(dy_des, axis=0) / self.dt  # T-1

      f_des = ddy_des - self.ay * (self.by * (self.goal - y_des[:-1]) - dy_des[:-1])  # T-1
      f_des = np.vstack((f_des, np.zeros((1, self.n_dmps))))  # T
      return f_des

  def gradient_force_v3(self,y_gt):
      f_des = self.imitate_path(y_gt)
      return self.force - f_des
      
if __name__  ==  "__main__":
  goal = np.array([0.1,0.3,0.3])
  goal_gt = np.array([0.2,0.2,0.4])
  start = np.array([-0.3,-0.2,-0.3])
  opti = None
  T = 49
  force_gt = np.random.uniform(0,1,size=(T,3))# * T * T
  force_gt[0:5,:] = -2000.0
  force_gt[5:10,:] = 2000.0
  force_gt[10:14,:] = -2000.0
  print("force_gt",force_gt.shape)
  d_test_gt = DDMP(opti,n_dmps=3,goal=goal_gt,start=start,force=force_gt,timesteps=T) 
  y_gt  = d_test_gt.rollout()[0]
  print("initial step",y_gt[0],"start",start)
  y_gt[15:,1] = y_gt[15,1]

  if 1:
    goal = np.copy(goal_gt)
    force = np.zeros_like(force_gt)
    d_test = DDMP(opti,n_dmps=3,goal=goal_gt, start=start,force=force,timesteps=T) 
    for i in range(100000000):
      #y_, dy_, ddy_ = d_test.rollout()
      grad_force = d_test.gradient_force_v3(y_gt)
      grad_goal = d_test.gradient_goal_v2(y_gt)
      force = force - .1* grad_force
      goal = goal - .2 * grad_goal
      d_test.set_force(force)
      d_test.set_goal(y_gt[-1])
      y_, dy_, ddy_ = d_test.rollout()
      print(np.sum((y_-y_gt)**2))
      if 1:
          fig = plt.figure(1)
          lenT = len(y_gt)
          plt.plot(np.arange(0,lenT),y_[:,1],color='red')
          plt.plot(np.arange(0,lenT),y_gt[:,1],color='green')
          fig.canvas.draw()
          images = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(480, 640, 3)
          cv2.imshow("Example",images)
          fig.canvas.figure.clf()
          cv2.waitKey(1)


