#!/bin/bash
#
#SBATCH --job-name=allen_two_p
#SBATCH --output=output_%A-%a.out 
#SBATCH --error=errors_%A-%a.err

#
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH --mem=1G

#SBATCH --time=02:00:00 #set for 2 hours

#
#SBATCH --gres=gpu:1  
#

echo $SLURM_ARRAY_TASK_ID

export PATH="$PATH:/nfs/nhome/live/ammarica/.local/bin"
cd /nfs/nhome/live/ammarica/mouse_connectivity_models
pipenv shell

python /nfs/nhome/live/ammarica/allendata/RSM_ventral-dorsal-model/analysis.py