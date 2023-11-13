#!/bin/bash
#SBATCH --job-name=sketch_loop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangcheng@idea.edu.cn
#SBATCH --partition=vanke
python sketch_loop.py 
