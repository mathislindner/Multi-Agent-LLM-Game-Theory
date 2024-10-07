#!/bin/bash
#SBATCH -n 4
#SBATCH --time=8:00
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=prisoners_dilemma_0
#SBATCH --output=prisoners_dilemma_0.out
#SBATCH --error=prisoners_dilemma_0.err

source .venv/bin/activate
python main.py --run