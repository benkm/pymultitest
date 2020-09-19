#!/bin/sh
#SBATCH --job-name=mega_array   # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk # Where to send mail	
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=40
#SBATCH --time=10:00:00             # Time limit hrs:min:sec
#SBATCH --output=GL_min_bayes_%A-%a.out    # Standard output and error log
#SBATCH --array=0-5                 # Array range

NO_CPUS=40

NO_GLS=14
NO_BBARS=10
NO_SAMPLES=40
NUM_TASKS=$(( $NO_GLS * $NO_BBARS * $NO_SAMPLES ))

ARRAY_SIZE=6
TASKS_PER_SCRIPT=$(( $NUM_TASKS / $ARRAY_SIZE ))

START=$(( $(( $NUM_TASKS * $SLURM_ARRAY_TASK_ID )) / $ARRAY_SIZE ))
END=$(( $(( $NUM_TASKS * ($SLURM_ARRAY_TASK_ID + 1) )) / $ARRAY_SIZE - 1 ))

echo ./worker_script.sh $START $END $NO_CPUS
