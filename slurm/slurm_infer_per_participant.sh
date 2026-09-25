#!/bin/bash -l
#SBATCH --job-name=infer_pp
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=infer_pp_%j.out
#SBATCH --error=infer_pp_%j.err

# ----- env setup -----
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

# NOTE: as of the post-process refactor, 02_postprocess.py no longer
# runs MCMC — it summarises the joint posterior already in hierarchical_fit.pkl
# into per-ppt marginals + PPC + per-ppt first-passage times. Runs in seconds, no GPU
# needed. The SLURM directives above still allocate a GPU for safety (so the
# JAX import works); a CPU-only partition would also work for this step.

cd ../analyses/hmm
echo "cwd: $(pwd)"

# Requires results/hierarchical_fit.pkl (from slurm_fit_hierarchical.sh)
if [ ! -f results/hierarchical_fit.pkl ]; then
    echo "ERROR: results/hierarchical_fit.pkl not found; run slurm_fit_hierarchical.sh first"
    exit 1
fi

python -u 02_postprocess.py
echo "02_postprocess.py done"
