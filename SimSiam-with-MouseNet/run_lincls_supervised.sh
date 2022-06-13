#!/bin/bash
#
#SBATCH --job-name=supervised_lincls_mousenet
#SBATCH --output=lincls_supervised_%A.out 

#
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G

#SBATCH --time=48:00:00 #set for 48 hours

#
#SBATCH --gres=gpu:a100:2  
#

source /usr/share/modules/init/bash
module load cuda/11.2

echo $SLURM_ARRAY_TASK_ID

export PATH="$PATH:/nfs/nhome/live/ammarica/.local/bin"
cd /nfs/nhome/live/ammarica/mouse_connectivity_models
pipenv shell

mkdir /nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_supervised_b_1024_lr_05

cd /nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_supervised_b_1024_lr_05

python3 /nfs/nhome/live/ammarica/SimSiam-with-MouseNet/main_lincls_supervised.py \
  -a mouse_net \
  -b 1024 \
  --lr 0.05 \
  --dist-url 'tcp://localhost:11111' --world-size 1 --rank 0 \
  /tmp/roman/imagenet