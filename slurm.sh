#!/bin/bash
#SBATCH -J Tabular        # Job name
#SBATCH -o /path/tabular.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#SBATCH --mail-user=YOUREMAIL
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#SBATCH --partition=gpuA40x4         # specify queue, if this doesn’t submit try gpu-shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --account=bbjr-delta-gpu
#SBATCH --mem=40G
#SBATCH --tasks=1              # Number of nodes, not cores (16 cores/node)
#SBATCH --nodes=1             # Total number of MPI tasks (if omitted, n=N); this is 1 for us
#SBATCH --time=4:00:00       # set maximum run time in H:m:S
#SBATCH --no-requeue     # don’t automatically requeue job if node fails, usually errors need to be inspected and debugged
# Load Anaconda module
module --ignore_cache load “anaconda3”
# Initialize conda
eval “$(conda shell.bash hook)”
# Activate the desired conda environment
conda activate flk_6
# Ensure the conda environment is activated properly
echo “Activated conda environment: $(conda info --envs | grep ‘*’ | awk ‘{print $1}’)”
# Run the training script
python path/train_cifar10.py --net res18 --n_epochs 200 --loss_eq mse --lr 2e-2