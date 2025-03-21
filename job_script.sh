#!/bin/bash

#SBATCH --job-name=job_%j
#SBATCH --output=logs/job_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=30-00:00:00
#SBATCH --mem=65536
#SBATCH --partition=waccamaw
#SBATCH --exclusive
#SBATCH --exclude=waccamaw01,waccamaw02

source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate testenv

CMD="python test.py"

echo "Running job ID $SLURM_JOB_ID on node $(hostname)"
echo "Executing command: $CMD"
eval $CMD
echo "Run finished"