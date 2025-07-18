#!/bin/bash -e
#SBATCH --job-name=fem
#SBATCH --time=144:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem=1GB 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-63
#SBATCH --account=uoa02799 


module load Python/3.10.5-gimkl-2022a
module load FreeFEM/4.15-foss-2023a
python3 run-script.py ${SLURM_ARRAY_TASK_ID}

## to output images on the cluster, prepend the output with "xvfb-run". e.g. "xvfb-run ./evolution"
