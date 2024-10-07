#!/bin/bash
$job_name = "prisoners_dilemma_0"
#SBATCH -n 4
#SBATCH --time=8:00
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=prisoners_dilemma_0
#SBATCH --output=prisoners_dilemma_0.out
#SBATCH --error=prisoners_dilemma_0.err

module load stack/2024-06 python_cuda/3.11.6