#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --mem=10G
#SBATCH --time=10:00:00          # walltime

for GL in 0.8 1.6 2.4 3.2 4 4.8 4.8 6.4 8 9.6 12.8 14.4 16 19.2 24 25.6 28.8 32
do
  python3 GL_min_bayes3.py $1 $2 $GL
done
