#!/bin/sh

#SBATCH -n 2
#SBATCH --mem-per-cpu=24G
#SBATCH --time=8:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g

#SBATCH --job-name=test_llama_0
#SBATCH --output=sbatch_out/test_llama.out
#SBATCH --error=sbatch_out/test_llama.err


source .venv/bin/activate

nvidia-smi

echo "testllama"

#pip install transformers
#pip install python-dotenv

python test_llama.py --model_id unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit