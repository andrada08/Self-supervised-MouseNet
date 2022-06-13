#!/bin/bash
#
#SBATCH --job-name=4gpu_unsupervised_mousenet
#SBATCH --output=array_%A.out 

#
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
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

python3 /nfs/nhome/live/ammarica/SimSiam-with-MouseNet/main_simsiam.py \
  -a mouse_net \
  -b 1024 \
  -j 16 \
  --epochs 100 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --fix-pred-lr \
  --save_dir '/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_b_1024_4_gpu/' \
  /tmp/roman/imagenet