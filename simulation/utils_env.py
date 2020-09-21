import numpy as np
import pybullet as p
#import matplotlib.pyplot as plt
import os
import torch
import shutil
import torch.autograd as Variable

#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D

def get_view(opt):
    def getview (width=600, height=600, look=[-0.05, -0.3, 0.0], dist=0.25, direct=[0.0, 0.0, 0.0]):
        cameraRandom = 0.0
        pitch = direct[0] + cameraRandom * np.random.uniform (-3, 3)
        yaw = direct[1] + cameraRandom * np.random.uniform (-3, 3)
        roll = direct[2] + cameraRandom * np.random.uniform (-3, 3)
        viewmatrix = p.computeViewMatrixFromYawPitchRoll (look, dist, yaw, pitch, roll, 2)

        fov = 40. + cameraRandom * np.random.uniform (-2, 2)
        aspect = float (width) / float (height)
        near = 0.01
        far = 10
        projmatrix = p.computeProjectionMatrixFOV (fov, aspect, near, far)

        return viewmatrix, projmatrix

    width = 640
    height = 480
    # TODO: Use config
    params_file = os.path.join("../configs/camera_parameters/params.npy")
    params = np.load(params_file)
    if opt.view_point == 'third':
        dist = params[5] + 0.3
        look = [params[3] - 0.4, -params[4], 0.0]
        direct = [params[0] + 90, params[2] + 180, params[1]]
    else:
        dist = params[5]
        look = [params[3], params[4], 0.0]
        direct = [params[0]+90,params[2],params[1]]
    view_matrix,proj_matrix = getview(width,height,look,dist,direct)
    return view_matrix,proj_matrix

def get_view_sim():
    def getview (width=600, height=600, look=[-0.05, -0.3, 0.0], dist=0.25, direct=[0.0, 0.0, 0.0]):
        cameraRandom = 0.0
        pitch = direct[0] + cameraRandom * np.random.uniform (-3, 3)
        yaw = direct[1] + cameraRandom * np.random.uniform (-3, 3)
        roll = direct[2] + cameraRandom * np.random.uniform (-3, 3)
        viewmatrix = p.computeViewMatrixFromYawPitchRoll (look, dist, yaw, pitch, roll, 2)

        fov = 40. + cameraRandom * np.random.uniform (-2, 2)
        aspect = float (width) / float (height)
        near = 0.01
        far = 10
        projmatrix = p.computeProjectionMatrixFOV (fov, aspect, near, far)

        return viewmatrix, projmatrix

    width = 640
    height = 480
    params = np.load('../../configs/camera_parameters/params.npy')
    # dist = params[5]
    dist = params[5]+0.3
    look = [params[3]-0.4, -params[4], 0.0]
    direct = [params[0]+90,params[2]+180,params[1]]
    view_matrix,proj_matrix = getview(width,height,look,dist,direct)
    return view_matrix,proj_matrix

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def point2traj(points=None,delta=0.01):
    traj = []
    last = points[0]
    for i in range(len(points)-1):
        now = points[i+1]
        diff = [x-y for x,y in zip(now,last)]
        dist = sum([x**2 for x in diff])**0.5
        n = int(dist/delta)
        for step in range(n):
            x = last[0] + step*delta*diff[0]/dist
            y = last[1] + step*delta*diff[1]/dist
            z = last[2] + step*delta*diff[2]/dist
            traj.append([x,y,z])
        last = now
    return traj

def get_gripper_pos(gripperOpen=1):
    """
    :param gripperOpen: 1 represents open, 0 represents close
    :return: the gripperPos
    """

    gripperLowerLimitList = [0] * 6
    gripperUpperLimitList = [0.81, -0.8, 0.81, -0.8, 0.8757, 0.8757]

    gripperPos = np.array (gripperUpperLimitList) * (1 - gripperOpen) + np.array (gripperLowerLimitList) * gripperOpen
    return gripperPos

def cut_frame(video_name,output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cmd = 'ffmpeg -i {} -vf scale=640:480 '.format(video_name)+'{}/%06d.jpg'.format(output_path)
    os.system(cmd)
    return output_path



def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

# use this to plot Ornstein Uhlenbeck random motion
def test_orn():
    ou = OrnsteinUhlenbeckActionNoise(1)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

class Visual:
    def __init__(self,test_id=0,epoch_id=0):
        self.test_id = test_id
        self.epoch_id = epoch_id
        self.log_file = '../../dataset/ddpg_log/test{}/epoch-{}.txt'\
            .format(self.test_id,self.epoch_id)
        self.traj_folder = safe_path('../../dataset/ddpg_log/test{}/traj/'.format(self.test_id))

    def update(self,epoch_id):
        self.epoch_id = epoch_id
        self.log_file = '../../dataset/ddpg_log/test{}/epoch-{}.txt' \
            .format (self.test_id, self.epoch_id)

    def get_xyz(self):
        self.points = []
        with open(self.log_file,'r') as reader:
            for line in reader.readlines():
                line = line.strip().split(':')
                if line[0]=='executed pos':
                    self.points.append(eval(line[1]))
        self.points = np.array(self.points)

    def show_xyz(self):
        self.get_xyz ()
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure ()
        ax = fig.gca (projection='3d')
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]
        ax.plot (x, y, z, label='parametric curve')
        ax.legend ()
        plt.savefig(os.path.join(self.traj_folder,'trajectory-{}.jpg'.format(self.epoch_id)))
        plt.cla()

def backup_code(script_path,log_path):
    log_path = os.path.join(log_path,'script_backup')
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    shutil.copytree(script_path,log_path)

def get_next_test_id():
    num = 0
    for file in os.listdir('../../dataset/ddpg_log/'):
        if 'test' in file:
            file = file.strip().replace('test','')
            num = max(num,int(file))
    return num+1

if __name__ == '__main__':
    agent = Visual(test_id=3)
    for epoch in range(300):
        agent.update(epoch+1)
        agent.show_xyz()
