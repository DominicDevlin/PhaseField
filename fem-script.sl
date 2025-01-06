#!/bin/bash -e
#SBATCH --job-name=CPM_evolution  
#SBATCH --time=12:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem=2GB 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-99  
#SBATCH --account=uoa02799         
#SBATCH --output=fem_sim_out-%j.out 
#SBATCH --error=fem_sim_err-%j.out 
#SBATCH --partition=milan


module load Python/3.10.5-gimkl-2022a
module load FreeFEM/4.15-foss-2023a
python3 oned-pyscript.py ${SLURM_ARRAY_TASK_ID}

## to output images on the cluster, prepend the output with "xvfb-run". e.g. "xvfb-run ./evolution"
