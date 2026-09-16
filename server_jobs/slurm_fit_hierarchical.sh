#!/bin/bash -l
#SBATCH --job-name=hier_fit
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=hier_fit_%j.out
#SBATCH --error=hier_fit_%j.err

# ----- env setup -----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "python: $(which python)"
python -c "import jax; print('jax:', jax.__version__, 'devices:', jax.devices())"

# JAX / numpy threading caps (matches existing slurm_ceiling.sh convention)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# ----- run config (scaled up vs local) -----
# Local defaults in fit_hierarchical.py are
#   N_OUTER=600, BURN=200, THIN=4, N_CHAINS=7, N_FFBS=1, N_SWAP=0
# Snellius "full" run: ~2x samples, 2x chains, 2x inner-MH depth.
# Chains differ only by PRNG seed (order inits are impossible: FFBS
# resamples O at the start of every sweep). (N_OUTER - BURN) need not be
# a multiple of THIN, but keep it so for round sample counts.
# Values already in the environment win (sbatch exports the caller's env by
# default), so `N_OUTER=20 BURN=5 THIN=1 N_CHAINS=2 sbatch ...` runs a smoke
# test instead of the full config; the defaults below apply otherwise.
export N_OUTER=${N_OUTER:-1200}
export BURN=${BURN:-400}
export THIN=${THIN:-4}
export N_CHAINS=${N_CHAINS:-14}
export N_FFBS=${N_FFBS:-1}
export N_SWAP=${N_SWAP:-0}
export CKPT_EVERY=${CKPT_EVERY:-100}
export RESUME=${RESUME:-0}

# JAX/CUDA performance knobs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

cd ../model/experiment_analysis
echo "cwd: $(pwd)"
echo "starting fit_hierarchical.py with N_OUTER=$N_OUTER BURN=$BURN THIN=$THIN N_CHAINS=$N_CHAINS RESUME=$RESUME"

python -u fit_hierarchical.py
echo "fit_hierarchical.py done"
