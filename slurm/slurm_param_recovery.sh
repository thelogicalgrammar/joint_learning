#!/bin/bash -l
#SBATCH --job-name=param_rec
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --output=param_rec_%A_%a.out
#SBATCH --error=param_rec_%A_%a.err

# Parameter recovery: one array task = one synthetic dataset refitted with
# 01_fit.py (see analyses/hmm/08_param_recovery.py).
# Before submitting, generate the datasets once (needs results/hierarchical_fit.pkl):
#   cd ../analyses/hmm && python 08_param_recovery.py simulate --mode $MODE --n_rep 8
# then, from slurm/:
#   MODE=prior sbatch slurm_param_recovery.sh          (array 0-7 = --n_rep 8)
#   MODE=null sbatch slurm_param_recovery.sh
#   MODE=posterior sbatch slurm_param_recovery.sh
# and afterwards:  python 08_param_recovery.py analyze --mode $MODE
# The fit config comes from the usual env vars; the recovery defaults are
# N_OUTER=1200 BURN=400 THIN=8 N_CHAINS=3 RESUME=1 (a re-submitted task
# continues from its checkpoints). Override on the sbatch line, e.g.
#   MODE=prior N_CHAINS=4 sbatch slurm_param_recovery.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "python: $(which python)"
# make the repository importable even if `pip install -e .` was not run in
# this env (the wrappers are submitted from slurm/, the root is one level up)
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -c "import jointlearn; print('jointlearn:', jointlearn.__file__)"
python -c "import jax; print('jax:', jax.__version__, 'devices:', jax.devices())"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

MODE=${MODE:-prior}
cd ../analyses/hmm
echo "cwd: $(pwd); mode $MODE, replicate $SLURM_ARRAY_TASK_ID"
python -u 08_param_recovery.py fit --mode "$MODE" --rep "$SLURM_ARRAY_TASK_ID"
echo "done"
