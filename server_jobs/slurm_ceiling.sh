#!/bin/bash -l
#SBATCH -c 16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 120:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "$(which python)"

cd ../
echo "Env activated, starting python job"s
python -m model.scripts.ceilingmodel