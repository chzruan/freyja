#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH -J DE2testL
#SBATCH -o ./dump/_ll_%J.dump
#SBATCH -e ./dump/_ll_%J.err
#SBATCH -p cosma8
#SBATCH -A dp203
#SBATCH -t 6:00:00

micromamba activate cosemu
cd /cosma8/data/dp203/dc-ruan1/freyja/scripts/xi_hh
python3 train_halo_bias.py