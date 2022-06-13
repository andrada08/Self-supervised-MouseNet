#!/bin/bash
#
#SBATCH --job-name=lincls_mousenet
#SBATCH --output=lincls_%A.out 

#
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=1G

#SBATCH --time=24:00:00 #set for 24 hours

#
#SBATCH --gres=gpu:a100:1  
#

source /usr/share/modules/init/bash
module load cuda/11.2

echo $SLURM_ARRAY_TASK_ID

export PATH="$PATH:/nfs/nhome/live/ammarica/.local/bin"
cd /nfs/nhome/live/ammarica/mouse_connectivity_models
pipenv shell

cd /nfs/nhome/live/ammarica/SimSiam-with-MouseNet

python3 /nfs/nhome/live/ammarica/SimSiam-with-MouseNet/main_lincls.py \
  -a mouse_net \
  -b 128 \
  --dist-url 'tcp://localhost:11101' --world-size 1 --rank 0 \
  --pretrained '/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/Checkpoints/checkpoint_0009.pth.tar' \
  /tmp/roman/imagenet