#!/bin/bash --login
#SBATCH -p gpuL 
#SBATCH -G 1
#SBATCH --output stdout-%j.log # stdout
#SBATCH --error stderr-%j.log #stderr
#SBATCH --ntasks-per-node 8
#SBATCH -t 0-2

# Clean modules
module purge

# Load Conda
module load apps/binapps/anaconda3/2024.10

# Load Env
source activate PyTorch-Env

# Enter working dir
cd ~/scratch/BReG-NeXt/codes/torch

# Run
python -u trainer.py
