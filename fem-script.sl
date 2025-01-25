#!/bin/bash -e
#SBATCH --job-name=fem
#SBATCH --time=12:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem=500MB 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-28 
#SBATCH --account=uoa02799         
#SBATCH --partition=milan


module load Python/3.10.5-gimkl-2022a
module load FreeFEM/4.15-foss-2023a
python3 run-script.py ${SLURM_ARRAY_TASK_ID}

## to output images on the cluster, prepend the output with "xvfb-run". e.g. "xvfb-run ./evolution"
