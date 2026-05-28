#!/bin/bash
#SBATCH --job-name=msp_bimodal_ser
#SBATCH --output=/home/lucas.ueda/slurm/msp_bimodal_%j.out
#SBATCH --error=/home/lucas.ueda/slurm/msp_bimodal_%j.err
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00  # Maximum 2 days as per your cluster limit
#SBATCH --mem=128G         # Increased memory for audio processing
#SBATCH --partition=l40s   # Selecting the h100 partition
#SBATCH --gres=gpu:1
#SBATCH --mail-user=l156368@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Miniconda and activate environment
source ~/miniconda3/bin/activate
conda activate alef  # Replace with your environment name

export WANDB_MODE=offline

# ----------------- EXPERIMENT 1: Fully Trainable Backbones -----------------
python src/main.py --config config/gated-bimodal-frozen-wavlm-qwen.yaml


TRAIN_EXIT_CODE=$?
echo "Training job exited with code: $TRAIN_EXIT_CODE"
