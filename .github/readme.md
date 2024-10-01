# Spinning Up - Renovated (WIP)

A renovation of "Spinning Up" reinforcement learning resources

[Link to original README](../readme.md)

### Updated installation instructions

(state: initially verified on Debian)

Linux

- Recommend using conda. To install conda, follow instructions on Anaconda website
- Install openmpi
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
- Create a conda environment on Python 3.11
```
conda create -n rl-spinningup python=3.11
```
- Activate conda environment
```
conda activate rl-spinningup
```
  - Run `which python` and `python -v` to verify.
- Install the project
```
pip install .
```
  - If there are errors such as lacking packages or unable to build wheel, try the following and rerun install:
```
conda install mpich-mpicc swig
```

https://stackoverflow.com/a/60990677 This might work for pygame installation problem
```
sudo apt-get install libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev
```
- once there are no errors, test the project
```
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```
- And test plotting as well as evaluation
```
python -m spinup.run test_policy /home/xxx/rl-spinningup/data/installtest/installtest_s0
```

### Mujoco

If having `fatal error: GL/osmesa.h: No such file or directory` errors:
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

`FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'`
```
sudo apt-get install patchelf
```
To solve AttributeError: 'mujoco._structs.MjData' object has no attribute 'solver_iter'

'GLIBCXX_3.4.30' not found
```
conda install -c conda-forge gcc=12.1.0
```


https://github.com/Farama-Foundation/Gymnasium/issues/749#issuecomment-1808355111
```
mujoco<3
```
works.


### Motivation of renovation and rough plans

Spinning Up is a great learning resource for reinforcement learning (RL). However, this project is in maintenance mode and is outdated, for example:

- The targeted python version == 3.6
- gym is no longer updated and new features are built in gymnasium
- The project still uses tensorflow 1.x

This fork plans to address these problems and make the project more up to date, with the following plans:

- [ ] Upgrade to newer python version (e.g., 3.11)
- [ ] Use gymnasium instead of gym
- [ ] Replace TF1 to a newer framework, such as TF2/Keras, JAX, FLAX
- [ ] Add new RL methods