#!/bin/sh

#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g

#SBATCH --job-name=prisoners_dilemma_0
#SBATCH --output=sbatch_out/prisoners_dilemma_1.out
#SBATCH --error=sbatch_out/prisoners_dilemma_1.err

echo "-> Loading modules required for build"

source .venv/bin/activate

nvidia-smi

echo "Running prisoners dilemma\n"

python main.py --model meta-llama/Llama-3.2-3B-Instruct