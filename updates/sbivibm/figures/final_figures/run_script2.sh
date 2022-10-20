#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --mem=32000
#SBATCH --time=0-2:00:00

python refined_posterior_compute_predictives.py