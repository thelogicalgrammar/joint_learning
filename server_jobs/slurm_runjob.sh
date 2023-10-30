#!/bin/bash
#SBATCH -c 4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 99:00:00

#### Tubingen setup
# module load 2021
# module load Python/3.9.5-GCCcore-10.3.0
# source ../../virtualenv/bin/activate

#### snellius UvA setup
module load 2022
module load Mamba/4.14.0-0

source deactivate
source activate joint_learning_fixed

echo "$(which python)"

cd ../

# launch 8 jobs in parallel, each using 4 cores
# for n in {1..8}; do
# 	python -m server_jobs.main --cores 4 --tune 4 --draws 4  --outputfile_append "$n" &
# done

echo "Venv activated, starting python job"

# python -m server_jobs.main --method metropolis --cores 4 --tune 1000 --draws 1500
python -m server_jobs.main --cores 4 --tune 1000 --target_accept 0.95 --draws 1000 --wo_prior_structure 'pooled'
