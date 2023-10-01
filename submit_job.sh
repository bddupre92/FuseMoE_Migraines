#!/bin/bash
#SBATCH --job-name=train-MulEHR
#SBATCH -t 10:00:00                  # estimated time
#SBATCH -n 6
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssaria1_gpu
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=xhan56@jhu.edu
#SBATCH --output=./slurm_files/slurm-train-MulEHR-xhan56.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-train-MulEHR-xhan56.err      # where to write slurm error

module load anaconda3
module load cuda
source activate MulEHR

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
sh run.sh
