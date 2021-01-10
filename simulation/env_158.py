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

from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('./Eval')
sys.path.append('./')

from env_57 import Engine57
from utils_env import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


class Engine158(Engine57):
    def __init__(self, worker_id, opti, p_id, taskId=5, maxSteps=15, n_dmps=3, cReward=True):
        super(Engine158,self).__init__(worker_id, opti, p_id, taskId=taskId, maxSteps=maxSteps, n_dmps=n_dmps, cReward=cReward)
        self.opti = opti

