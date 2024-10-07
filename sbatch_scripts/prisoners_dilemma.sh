#!/bin/bash
#SBATCH -n 1
#SBATCH --time=8:00
#SBATCH --mem-per-cpu=20000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=prisoners_dilemma_0
#SBATCH --output=sbatch_out/prisoners_dilemma_0.out
#SBATCH --error=sbatch_out/prisoners_dilemma_0.err

source .venv/bin/activate
python main.py --run