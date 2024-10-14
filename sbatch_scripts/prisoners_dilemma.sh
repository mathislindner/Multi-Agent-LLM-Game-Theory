#!/bin/sh
#SBATCH -n 2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=8:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g

#SBATCH --job-name=prisoners_dilemma_2
#SBATCH --output=sbatch_out/prisoners_dilemma_2.out
#SBATCH --error=sbatch_out/prisoners_dilemma_2.err

echo "-> Loading modules required for build"

source .venv/bin/activate

nvidia-smi

echo "Running prisoners dilemma\n"

export HF_HOME=/cluster/scratch/mlindner/cache/
python main.py --model_id meta-llama/Llama-3.1-8B-Instruct