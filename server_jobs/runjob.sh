#!/bin/bash
#SBATCH -n 16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 1:00:00

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
source ../../venv/bin/activate

python ../model/scripts/weight_model_fit.py
