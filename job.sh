#!/bin/bash

#SBATCH --job-name LSTM_classencoded_hidden1024_dropout0.5
#SBATCH --gres gpu:1
#SBATCH --partition gpu
#SBATCH --time 0-10:00
#SBATCH --mem=100GB
#SBATCH --output output_LSTM_classencoded_hidden1024_dropout0.5

# Print some information about the job
echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is $SLURM_JOB_ID"
echo "Number of visible devices = $CUDA_VISIBLE_DEVICES"
echo
echo "This job runs on the following machines:"
echo "$SLURM_JOB_NODELIST" | uniq
echo

module load CUDA
module load cuDNN

# Run the Python script
srun python train.py --config config.json
