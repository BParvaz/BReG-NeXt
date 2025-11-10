#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 0-1       # 1 hour wallclock time limit (max permitted is 7-0, i.e. 7 days)

# generic jobscript for:
# installing conda
# loading conda env
# installing dependencies and version

module purge
module load apps/binapps/anaconda3/2024.10  # Python 3.12.7


# Python 3.10 environment
conda create -n PyTorch-Env python=3.10 -y

# Activate Environment
source activate PyTorch-Env

# pytorch-lightning is not entirely necessary for now
# but i intend to implement cli args using lightningCLI
conda install pytorch torchvision pytorch-lightning -c pytorch-nightly -c nvidia -c conda-forge -y

# tfrecord is a tensorflow binary, not independent on conda
# needs to be installed using pip itself
pip install tfrecord -y


conda list
# just for good measure and debugging, not necessary

source deactivate