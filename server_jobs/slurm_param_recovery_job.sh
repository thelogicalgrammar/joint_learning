#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --time 10:00:00

# This script should be run as an array script, e.g.,
# sbatch --array=0-50 slurm_param_recovery_job.sh

module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate
conda activate LoT_recovery

cd ../

echo "Starting python job"
python -m server_jobs.parameter_recovery_simulations --n $SLURM_ARRAY_TASK_ID
