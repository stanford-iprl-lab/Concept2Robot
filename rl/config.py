"""
the whole project only has one absolute path, which is the project_root parameter in opt
quthor: Qiang Zhang
time: 7-28-2019
"""

import argparse
import torch
import pathlib

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# reinforcement learning part hyper parameters
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--update_time', default=1, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1e6, type=int) # replay buffer size
parser.add_argument('--start_train', default=50, type=int) # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=16, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)


# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render',type=str2bool, nargs='?',const=True, default=False) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', type=str2bool, nargs='?',const=True, default=False) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.001, type=float)
parser.add_argument('--noise_level', default=0.7, type=float)
parser.add_argument('--noise_clip', default=0.03, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.001, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)


# environment part hyper parameters
# environment part hyper parameters
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.absolute())
parser.add_argument('--project_dir',  default=ROOT_DIR, type=str) # project root path

parser.add_argument('--project_root',  default='/juno/u/lins2/ConceptManipulation/simulation', type=str) # project root path
# parser.add_argument('--project_root',  default='/juno/u/qiangzhang/system/gamma-robot/', type=str) # project root path
parser.add_argument('--test_id',  default=1113, type=int) #  1000+ means debug
parser.add_argument('--video_id',  default=6, type=int) #

parser.add_argument('--object_id',  default='nut', type=str) #
parser.add_argument('--observation',  default='before_cnn', type=str) # after_cnn or joint_pose or end_pos or before_cnn or others
parser.add_argument('--view_point',  default='first', type=str) # first or third
parser.add_argument('--rand_start',  default='fixed', type=str) # rand or two or others
parser.add_argument('--rand_box',  default='fixed', type=str) # rand or two or others

parser.add_argument('--axis_limit_x',  default='[0.04,0.6]', type=str) #
parser.add_argument('--axis_limit_y',  default='[-0.40,0.25]', type=str) #
parser.add_argument('--axis_limit_z',  default='[0.26,0.7]', type=str) #
parser.add_argument('--img_w',  default=160, type=int) #
parser.add_argument('--img_h',  default=120, type=int) #

parser.add_argument('--obj_away_loss',  default=True, type=int) #
parser.add_argument('--away_reward',  default=0, type=float) #
parser.add_argument('--reward_diff',  default=True, type=int) #
parser.add_argument('--out_reward',  default=-10, type=float) #

parser.add_argument('--end_distance',  default=0.20, type=float) #
parser.add_argument('--each_action_lim',  default=0.03, type=float) #

parser.add_argument('--rand',  type=str2bool, nargs='?',const=True, default=False) # rand or two or others

# video prediction part hyper parameters
parser.add_argument('--action_id',  default=106, type=int) #
parser.add_argument('--cut_frame_num',  default=20, type=int) #
parser.add_argument('--give_reward_num',  default=1, type=int) #
parser.add_argument('--video_reward',  default=True, type=bool) #
parser.add_argument('--load_video_pred',  default=None, type=object) #
parser.add_argument('--add_mask',  default=True, type=int) #
parser.add_argument('--prob_softmax',  default=False, type=int) #
parser.add_argument('--merge_class',  default=True, type=int) #


# environment action using DMP part hyperparameters
parser.add_argument('--actions_root', default='/scr1/system/beta-robot/dataset/actions', type=str) #

# environment action using DMP part hyperparameters
parser.add_argument('--use_cycle',  default=False, type=int) #
parser.add_argument('--load_cycle', default=None, type=object) #

# environment objects urdf root
parser.add_argument('--urdf_root',default="/juno/u/lins2/ConceptManipulation/simulation",type=str)
parser.add_argument('--recordedVideo_root',default="/juno/u/lins2/ConceptManipulation/Data/recordedVideos",type=str)
parser.add_argument('--task_id',default=5,type=int)
parser.add_argument('--master',type=str2bool, nargs='?',const=True, default=False)
parser.add_argument('--start_learning',default=500,type=int)
parser.add_argument('--gui',type=str2bool, nargs='?',const=True, default=False,help="using gui.")
parser.add_argument('--test',type=str2bool, nargs='?',const=True, default=False,help="add coupling term.")
parser.add_argument('--restore',type=str2bool, nargs='?',const=True, default=False,help="add coupling term.")
parser.add_argument('--restore_step',default=0,type=int)
parser.add_argument('--restore_master',default=False,type=bool)
parser.add_argument('--mem_dir',default='mem',type=str)
parser.add_argument('--rotation_max_action',default=0.0,type=float)
parser.add_argument('--explore_var',default=0.5,type=float)
parser.add_argument('--force_term',type=str2bool, nargs='?',const=True, default=False,help="add force term.")
parser.add_argument('--exp_name',default="RL_No_Force_test",type=str)
parser.add_argument('--restore_path',default=None,type=str)
parser.add_argument('--coupling_term',type=str2bool, nargs='?',const=True, default=False,help="add coupling term.")
parser.add_argument('--traj_lr',type=float,default=1e-5)
parser.add_argument('--traj_timesteps',type=int,default=49)
parser.add_argument('--dmp_timesteps',type=int,default=49)
parser.add_argument('--use_dmp',type=str2bool,default=True)
parser.add_argument('--dmp_time_param', default='t', choices=['t','s']) #
parser.add_argument('--use_cem',type=str2bool,default=False)
parser.add_argument('--only_force_term',type=str2bool,default=False)
parser.add_argument('--only_coupling_term',type=str2bool,default=False)
parser.add_argument('--start_learning_coupling',type=int,default=2000)
parser.add_argument('--videoReward',type=str2bool,default=True)
parser.add_argument('--classifier',default='video',choices=['video','image','tsm_video','trn_video'],type=str)
parser.add_argument('--recordGif',type=str2bool,default=False)
parser.add_argument('--action_penalty',type=float,default=1.0)
parser.add_argument('--test_freq',type=int,default=1000)
parser.add_argument('--max_ep_test',type=int,default=100)
parser.add_argument('--TDI',type=int,default=0)

opti = parser.parse_args()
