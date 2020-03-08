#!/bin/bash

#SBATCH --job-name test_train
#SBATCH --gres gpu:1
#SBATCH --partition gpu_veryshort
#SBATCH --time 0-01:00
#SBATCH --mem=64GB
#SBATCH --output test_output

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
# module load CUDA/8.0.44
# module load cuDNN/5.1-CUDA-8.0.44
# module load languages/anaconda3/2019.10-3.7.4-tflow-2.0.0

# Run the Python script
# srun cd ~/great-ape-behaviour-detector
# srun cd ..
srun pwd
# srun conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# conda list
# srun echo hi
# srun conda init bash
# srun conda activate conda-env
srun python main.py --name sbatch_test
