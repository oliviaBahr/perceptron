#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=amilan
#SBATCH --out=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

module purge
module load gcc/11.2.0
module load mambaforge
mamba activate perc-soup

cd "/projects/$USER/perceptron"

python src/run_hyperparam_sweep.py
