#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=8000
#SBATCH --time=0-00:15:00

cd $3
python run_sims_for_parameters.py $1 $2 $4 
