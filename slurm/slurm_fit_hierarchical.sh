#!/bin/bash -l
#SBATCH --job-name=hier_fit
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=hier_fit_%j.out
#SBATCH --error=hier_fit_%j.err

# ----- env setup -----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "python: $(which python)"
# make the repository importable even if `pip install -e .` was not run in
# this env (the wrappers are submitted from slurm/, the root is one level up)
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -c "import jointlearn; print('jointlearn:', jointlearn.__file__)"
python -c "import jax; print('jax:', jax.__version__, 'devices:', jax.devices())"

# JAX / numpy threading caps (matches existing slurm_ceiling.sh convention)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# ----- run config -----
# Defaults = the reported fit (also the sampler.py defaults): HMC-within-Gibbs,
# 8 chains x (1000 warm-up + 6000 iterations, thin 20) = 2400 pooled draws,
# trajectory length 3.3 x U[0.5, 1.5], chain k seeded SEED0 + k. The reported
# fit was run on a laptop RTX 3050 Ti (~15 min per chain), 2026-11-10.
# 01_fit.py runs each chain in its own process, one after the other. Values
# already in the environment win, so
# `N_WARMUP=30 N_SAMPLES=40 THIN=4 N_CHAINS=2 sbatch ...` runs a smoke test.
export N_WARMUP=${N_WARMUP:-1000}
export N_SAMPLES=${N_SAMPLES:-6000}
export THIN=${THIN:-20}
export N_CHAINS=${N_CHAINS:-8}
export TRAJ=${TRAJ:-3.3}
export SEED0=${SEED0:-20000}
export RESUME=${RESUME:-0}

# JAX/CUDA performance knobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

cd ../analyses/hmm
echo "cwd: $(pwd)"
echo "starting 01_fit.py with N_WARMUP=$N_WARMUP N_SAMPLES=$N_SAMPLES THIN=$THIN N_CHAINS=$N_CHAINS TRAJ=$TRAJ SEED0=$SEED0 RESUME=$RESUME"

python -u 01_fit.py
echo "01_fit.py done"
