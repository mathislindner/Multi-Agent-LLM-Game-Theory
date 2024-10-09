#!/bin/sh

#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g

#SBATCH --job-name=prisoners_dilemma_0
#SBATCH --output=sbatch_out/prisoners_dilemma_0.out
#SBATCH --error=sbatch_out/prisoners_dilemma_0.err

echo "-> Loading modules required for build"

#module load stack/2024-06
#module load python/3.11.6 #python_gpu/3.11.6

#module load stack/2024-06 python_gpu/3.11.6
#python -m venv --system-site-packages .venv

source .venv/bin/activate

nvidia-smi

echo "Running prisoners dilemma\n"

#pip install transformers
#pip install python-dotenv

python main.py --model meta-llama/Llama-3.2-3B-Instruct