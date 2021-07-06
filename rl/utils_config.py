import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR,'../')

def load_args():
    parser = argparse.ArgumentParser(description='concept2robot')
    parser.add_argument('--project_dir', default=PROJECT_DIR, type=str, help='project root directory')
    parser.add_argument('--use_cem', action='store_true')
    parser.add_argument('--comment',default="", type=str)

    #### model specification
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

    ### experiment specification
    parser.add_argument('--classifier', default='video', type=str, choices=['video', 'image'])
    parser.add_argument('--max_ep', default=500000, type=int, help="maximum episode in the training stage")
    parser.add_argument('--max_action', default=0.5, type=float, help='maximum action in translation')
    parser.add_argument('--rotation_max_action', default=0.1, type=float, help='maximum action in rotation')

    ## training specification
    parser.add_argument('--a_lr', default=1e-5, type=float, help='the learning rate of the actor')
    parser.add_argument('--c_lr', default=5e-5, type=float, help='the learning rate of the critic')

    parser.add_argument('--explore_var', default=0.5, type=float, help='the exploring variable')
    parser.add_argument('--explore_decay', default=0.9999, type=float, help='the exploring variable')
    parser.add_argument('--start_learning_episode', default=2000, type=float, help='start learning step')
    parser.add_argument('--saving_model_freq', default=1000, type=int, help='how often to save the current trained model weight')
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--mem_capacity', default=50000, type=int, help='the capacity of the reply buffer')
    parser.add_argument('--video_reward', action='store_false', help="use video classification as the reward")
    parser.add_argument('--gt_reward', action='store_true',help="use ground truth reward")
    parser.add_argument('--action_penalty', default=0.1, type=float, help="action penalty")

    ## testing specification
    parser.add_argument('--restore_old',action='store_true')
    parser.add_argument('--max_ep_test', default=100, type=int, help='maximum episode in the test stage')
    parser.add_argument('--restore_episode', default=0, type=int, help='restore step')
    parser.add_argument('--restore_path', default=None, type=str, help='directory of the pretrained model')
    parser.add_argument('--restore_episode_goal', default=0, type=int, help='restore step')
    parser.add_argument('--restore_path_goal', default=None, type=str, help='directory of the pretrained model')

    ## demonstration specification
    parser.add_argument('--demonstration_dir', default="../demonstration_dir",type=str,help="directory which contains the demonstration data")
    parser.add_argument('--max_ep_demon', default=2000, type=int,
                        help='the number of episode generated in the demonstration')

    ## imitation specification
    parser.add_argument('--m_lr', default=1e-5, type=float, help='the learning rate of the trajectory')
    parser.add_argument('--max_ep_imitation', default=30, type=int, help='maximum episode in the imitation stage')
    parser.add_argument('--max_iteration', default=1000000, type=int, help="max number of iteration in imitation learning")
    parser.add_argument('--saving_model_freq_imitation', default=10000, type=int,
                        help='how often to save the current trained model weight')

    ## feedback specification
    parser.add_argument('--feedback_term', action='store_true', help="use feedback term")
    parser.add_argument('--a_f_lr', default=1e-4, type=float, help='the learning rate of the actor_feedback')
    parser.add_argument('--c_f_lr', default=1e-3, type=float, help='the learning rate of the critic_feedback')

    parser.add_argument('--start_learning_timestep', default=2000, type=float, help='start learning step')
    parser.add_argument('--max_feedback_action', default=0.015, type=float, help='maximum action in translation')
    parser.add_argument('--rotation_max_feedback_action', default=0.0, type=float, help='maximum action in rotation')
    parser.add_argument('--discount', default=0.9, type=float, help='maximum action in rotation')
    parser.add_argument('--max_ep_feedback_test', default=30, type=int, help='maximum episode in the test stage')
    parser.add_argument('--stack_num', default=4, type=int)
    parser.add_argument('--mem_feedback_capacity', default=30000, type=int, help='the capacity of the reply buffer') #### make sure mem_feedback_capacity % stack_num == 0
    parser.add_argument('--restore_feedback_episode',default=0,type=int)
    parser.add_argument('--restore_feedback_path',default="",type=str)

    ##feedback imitation specification
    parser.add_argument('--c_m_lr', default=1e-5, type=float, help='the learning rate of the trajectory')
    parser.add_argument('--max_ep_imitation_feedback', default=30, type=int, help='maximum episode in the imitation stage')
    parser.add_argument('--max_iteration_feedback', default=1000000, type=int, help="max number of iteration in imitation learning")

    ### environment or task specification
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--recordGif', action='store_true')
    parser.add_argument('--gif_dir', default="../gif_dir", type=str, help="directory of generated gif")
    parser.add_argument('--log_dir', default="../log_dir", type=str)
    parser.add_argument('--save_dir', default="../save_dir", type=str)

    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--task_id', default=0, type=int, help='the task id')
    parser.add_argument('--exp_name', default='without_force', type=str)
    parser.add_argument('--method', default='rl', type=str)
    parser.add_argument('--wid', default=5, type=int, help='wid in pybullet')
    parser.add_argument('--view_point', default='first', type=str, choices=['first', 'third'],
                        help='viewpoint of the camera')
    parser.add_argument('--stage', default='train', type=str, help='which stage to execute')
    args = parser.parse_args()
    return args
