# Concept2Robot
We aim to endow a robot with the ability to learn manipulation concepts that link natural language instructions to
motor skills. Our goal is to learn a single multi-task policy that takes as input a natural language instruction and an image of
the initial scene and outputs a robot motion trajectory to achieve the specified task. This policy has to generalize over different instructions and environments. Our insight is that we can approach this problem through Learning from Demonstration by leveraging large-scale video datasets of humans performing manipulation
actions. Thereby, we avoid more time-consuming processes such as teleoperation or kinesthetic teaching. We also avoid having to
manually design task-specific rewards. We propose a two-stage learning process where we first learn single-task policies through
reinforcement learning. The reward is provided by scoring how well the robot visually appears to perform the task. This score is
given by a video-based action classifier trained on a large-scale human activity dataset. In the second stage, we train a multi-task
policy through imitation learning to imitate all the single-task policies. In extensive simulation experiments, we show that the
multi-task policy learns to perform a large percentage of the 78 different manipulation tasks on which it was trained. The tasks
are of greater variety and complexity than previously considered robot manipulation tasks. We show that the policy generalizes
over variations of the environment. We also show examples of successful generalization over novel but similar instructions.

[Project Webpage](https://sites.google.com/view/concept2robot)

## Installation

1. Initialize repository
```
git submodule init && git submodule update
```

2. Compile bullet
```
cd external/bullet3; bash build_cmake_pybullet_double.sh
```

3. Install ffmpeg
```
sudo apt install autoconf automake build-essential cmake libass-dev libfreetype6-dev libjpeg-dev libtheora-dev libtool libvorbis-dev libx264-dev pkg-config wget yasm zlib1g-dev
wget https://www.ffmpeg.org/releases/ffmpeg-4.2.1.tar.xz
tar -xf ffmpeg-4.2.1.tar.xz
cd ffmpeg-4.2.1
./configure --disable-static --enable-shared --disable-doc
make
sudo make install
```

4. Create and Initialize Conda Environment
```
conda env create -f environment.yml
conda activate concept2robot
```

5. Download [data](http://download.cs.stanford.edu/juno/Concept2Robot/data.zip), [models](http://download.cs.stanford.edu/juno/Concept2Robot/models.zip) folders into `ConceptManipulation` directory.

6. Run the code
```
cd rl; bash train_5.sh
```

If you think our work is useful, please consider citing use with
```
@inproceedings{lin2020concept,
 title={Concept2Robot: Learning Manipulation Concepts from Instructions and Human Demonstrations},
 author={Shao, Lin and Migimatsu, Toki and Zhang, Qiang and Yang, Karen and Bohg, Jeannette},
 booktitle={Proceedings of Robotics: Science and Systems (RSS)},
 year={2020},
 month={July}}
```
