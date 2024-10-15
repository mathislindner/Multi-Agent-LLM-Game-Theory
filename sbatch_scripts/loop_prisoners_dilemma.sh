#!/bin/sh
#SBATCH -n 2
#SBATCH --mem-per-cpu=64G
#SBATCH --time=30:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g

#SBATCH --job-name=prisoners_dilemma_2
#SBATCH --output=sbatch_out/prisoners_loop.out
#SBATCH --error=sbatch_out/prisoners_loop.err

echo "-> Loading modules required for build"

source .venv/bin/activate

nvidia-smi

echo "Running prisoners dilemma\n"

export HF_HOME=/cluster/scratch/mlindner/cache/
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona selfish --agent_2_persona selfish
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona selfish --agent_2_persona altruistic
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona altruistic --agent_2_persona altruistic
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona no_persona --agent_2_persona no_persona
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona no_persona --agent_2_persona selfish
python main_loop.py --model_id meta-llama/Llama-3.1-8B-Instruct --rounds 20 --agent_1_persona no_persona --agent_2_persona altruistic