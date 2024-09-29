from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==3',
        'gymnasium[atari,box2d,classic_control]~=0.29.0',
        'gymnasium[mujoco]',
        'mujoco-py<2.2,>=2.1',
        'ipython',
        'ipykernel',
        'joblib',
        'matplotlib==3.9.2',
        'mpi4py',
        'numpy',
        'opencv-python==4.10.*',
        'opencv-python-headless==4.10.*',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.13.2',
        # 'tensorflow>=1.8.0,<2.0',
        'torch==2.4.1',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
