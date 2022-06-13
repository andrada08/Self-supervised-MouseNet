#!/bin/bash
#
#SBATCH --job-name=005_lincls_mousenet
#SBATCH --output=lincls_%A.out 

#
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G

#SBATCH --time=48:00:00 #set for 48 hours

#
#SBATCH --gres=gpu:a100:1  
#

source /usr/share/modules/init/bash
module load cuda/11.2

echo $SLURM_ARRAY_TASK_ID

export PATH="$PATH:/nfs/nhome/live/ammarica/.local/bin"
cd /nfs/nhome/live/ammarica/mouse_connectivity_models
pipenv shell

cd /nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_4_gpu_b_1024_lr_005

python3 /nfs/nhome/live/ammarica/SimSiam-with-MouseNet/main_lincls.py \
  -a mouse_net \
  -b 1024 \
  --dist-url 'tcp://localhost:11101' --world-size 1 --rank 0 \
  --pretrained '/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_4_gpu_b_1024_lr_005/checkpoint_0099.pth.tar' \
  /tmp/roman/imagenet