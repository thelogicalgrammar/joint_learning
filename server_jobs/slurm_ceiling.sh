#!/bin/bash
#SBATCH -c 16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 120:00:00

#### Tubingen setup
# module load 2021
# module load Python/3.9.5-GCCcore-10.3.0
# source ../../virtualenv/bin/activate

#### Snellius UvA setup
module load 2022
module load Mamba/4.14.0-0

source deactivate
source activate joint_learning_fixed

echo "$(which python)"

cd ../

echo "Venv activated, starting python job"

python -m server_jobs.main 