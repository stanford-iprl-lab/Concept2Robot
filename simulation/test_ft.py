import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pybullet as p
import pybullet_data

id = p.connect(p.GUI)
# gcomp_id = p.connect(p.SHARED_MEMORY)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pth = os.path.join(os.path.dirname(__file__), '../resources/urdf/franka_panda/panda_robotiq.urdf')
pth_plane = os.path.join(os.path.dirname(__file__), '../resources/urdf/plane.urdf')
print(pth)
robotId = p.loadURDF(pth, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION and p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
p.resetBasePositionAndOrientation(robotId, [0, 0, 0], [0, 0, 0, 1])
p.loadURDF(pth_plane)

# print("mass of linkA = 1kg, linkB = 1kg, total mass = 2kg")

ee_index = 18
# 18 or 19 should work here (finger pads)
tip_idx = 18

num_joints = p.getNumJoints(robotId)
num_dof = 0
for i in range(num_joints):
    num_dof += int(p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED)

print("Joints / dof:", num_joints, num_dof)
#by default, joint motors are enabled, maintaining zero velocity
p.setJointMotorControl2(robotId, ee_index, p.VELOCITY_CONTROL, 0, 0, 0)

p.setGravity(0, 0, -10)
p.stepSimulation()
print("joint state without sensor")

print(p.getJointState(0, ee_index))
p.enableJointForceTorqueSensor(robotId, ee_index)
p.stepSimulation()
print("joint state with force/torque sensor, gravity [0,0,-10]")
print(p.getJointState(0, ee_index))
p.setGravity(0, 0, 0)
p.stepSimulation()
print("joint state with force/torque sensor, no gravity")
print(p.getJointState(0, ee_index))


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
xdata, ydata, ydata_g = [], [], []
lns, lns_g = [], []

for i in range(6):
    ln, = axes[i // 3, i % 3].plot([], [], label='raw')
    ln2, = axes[i // 3, i % 3].plot([], [], label='comp')
    if i // 3 == 0:
        title = 'force_'
    else:
        title = 'torque_'
    title += "%s" % (['x', 'y', 'z'][i % 3])
    cpts = p.getContactPoints(bodyA=robotId, linkIndexA=18)  # 18, 19 are the finger pads
    axes[i // 3, i % 3].set_title(title)
    axes[i // 3, i % 3].legend()
    lns.append(ln)
    lns_g.append(ln2)
    ydata.append([])
    ydata_g.append([])
p.setGravity(0, 0, -10)
# p.setGravity(0, 0, -10, physicsClientId=gcomp_id)

def init():
    return lns

def update(frame):
    p.stepSimulation()
    # ee orn
    ee_orn = p.getLinkState(robotId, ee_index)[1]
    g_orn = p.getLinkState(robotId, tip_idx)[1]

    ee_orn_mat = np.asarray(p.getMatrixFromQuaternion(ee_orn)).reshape((3,3))
    g_orn_mat = np.asarray(p.getMatrixFromQuaternion(g_orn)).reshape((3,3))

    state = list(p.getJointState(robotId, ee_index)[2])
    state[:3] = (ee_orn_mat @ np.asarray(state[:3])).tolist()
    state[3:] = (ee_orn_mat @ np.asarray(state[3:])).tolist()
    out = p.getJointStates(robotId, range(num_dof))
    q = [o[0] for o in out]
    qdot = [o[1] for o in out]
    # pad for other degrees of freedom
    q += (num_dof - len(out)) * [0]
    qdot += (num_dof - len(out)) * [0]
    eeJt, eeJr = p.calculateJacobian(robotId, ee_index, [0,0,0], objPositions=q, objVelocities=qdot, objAccelerations=[0]*num_dof)
    eeJt = np.asarray(eeJt)
    eeJr = np.asarray(eeJr)

    # compute jacobian for gripper (tip)
    gJt, gJr = p.calculateJacobian(robotId, tip_idx, [0, 0, 0], objPositions=q, objVelocities=qdot,
                                     objAccelerations=[0] * num_dof)
    gJt = np.asarray(gJt)
    gJr = np.asarray(gJr)

    force_ee = np.asarray(state[:3])
    torque_ee = np.asarray(state[3:])

    # computing the force at the tip using relative jacobian
    force_g = gJt @ (eeJt.T @ force_ee)
    torque_g = gJr @ (eeJr.T @ torque_ee)

    state_g = force_g.tolist() + torque_g.tolist()

    xdata.append(frame)

    for i in range(2):
        for j in range(3):
            y = state[i * 3 + j]
            y_g = state_g[i * 3 + j]

            ylist = ydata[i * 3 + j]
            ylist_g = ydata_g[i * 3 + j]

            ylist.append(y)
            ylist_g.append(y_g)
            axes[i, j].set_xlim(frame-100, frame)
            axes[i, j].set_ylim(min(ylist[-100:] + ylist_g[-100:]) - 1, max(ylist[-100:] + ylist_g[-100:]) + 1)
            lns[i * 3 + j].set_data(xdata, ylist)
            lns_g[i * 3 + j].set_data(xdata, ylist_g)
    return lns

ani = FuncAnimation(fig, update, interval=50,
                    init_func=init, blit=False)
plt.show()



p.disconnect()