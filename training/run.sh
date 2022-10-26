#!/bin/bash
#SBATCH -n 64
#SBATCH -t 48:00:00
#SBATCH -q css
#Other SBATCH commands go here

#Activating conda
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate training

#Your appication commands go here
python main.py

