#!/bin/bash
#SBATCH -c 16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 120:00:00

#### Snellius UvA setup
module load 2022
conda activate pymc523

echo "$(which python)"

cd ../

echo "Env activated, starting python job"

python -m model.scripts.ceilingmodel