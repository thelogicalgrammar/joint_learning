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
module module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0 

source deactivate
source activate pymc523

echo "$(which python)"

cd ../

echo "Env activated, starting python job"

python -m model.scripts.ceilingmodel