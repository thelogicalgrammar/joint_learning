#!/bin/bash -l
#SBATCH -c 16
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 120:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "$(which python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

python - <<'PY'
import os, sys, multiprocessing as mp
print("Python:", sys.version.split()[0])
print("Start method:", mp.get_start_method())
print("Conda env:", os.getenv("CONDA_DEFAULT_ENV"))
print("OMP:", os.getenv("OMP_NUM_THREADS"))
print("MKL:", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS:", os.getenv("OPENBLAS_NUM_THREADS"))
PY

cd ../
echo "Env activated, starting python job"s
python -m model.scripts.ceilingmodel