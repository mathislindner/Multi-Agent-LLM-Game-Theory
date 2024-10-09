#!/bin/sh

#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g

#SBATCH --job-name=test_llama_0
#SBATCH --output=sbatch_out/test_llama.out
#SBATCH --error=sbatch_out/test_llama.err

echo "-> Loading modules required for build"

module purge
module load stack/2024-06
#module load python_gpu/3.11.6 #python_gpu/3.11.6

source .venv/bin/activate

nvidia-smi

echo "testllama"

#pip install transformers
#pip install python-dotenv

python test_llama.py