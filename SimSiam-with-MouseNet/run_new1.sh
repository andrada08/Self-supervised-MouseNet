#!/bin/bash
#
#SBATCH --job-name=lr_1_unsupervised_mousenet
#SBATCH --output=array_%A.out 

#
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --exclude=gpu-sr670-20
#SBATCH --mem=40G

#SBATCH --time=120:00:00 #set for 5 days

#
#SBATCH --gres=gpu:a100:4
#

source /usr/share/modules/init/bash
module load cuda/11.2

echo $SLURM_ARRAY_TASK_ID

export PATH="$PATH:/nfs/nhome/live/ammarica/.local/bin"
cd /nfs/nhome/live/ammarica/mouse_connectivity_models
pipenv shell

cd /nfs/nhome/live/ammarica/SimSiam-with-MouseNet

mkdir /nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_4_gpu_b_1024_lr_005/

python3 -u /nfs/nhome/live/ammarica/SimSiam-with-MouseNet/main_simsiam.py \
  -a mouse_net \
  -b 1024 \
  -j 16 \
  --epochs 100 \
  --lr 0.005 \
  --dist-url 'tcp://localhost:10101' --multiprocessing-distributed --world-size 1 --rank 0 --fix-pred-lr \
  --save_dir '/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_4_gpu_b_1024_lr_005/' \
  /tmp/roman/imagenet