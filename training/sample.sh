#!/bin/bash
#SBATCH -n 28
##SBATCH -t 48:00:00
##SBATCH -p nvidia
##SBATCH --gres=gpu:1


#Other SBATCH commands go here

#Activating conda
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate training

#Your appication commands go here
python sample.py

