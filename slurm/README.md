# Running the HMM fit on Snellius

Two SLURM scripts, both submitted from this folder (they `cd ../analyses/hmm`
and put the repository root on `PYTHONPATH`):

- `slurm_fit_hierarchical.sh` — `01_fit.py` on one A100 at the reported
  configuration (4800 outer iterations, burn-in 1200, thin 12, 14 chains; the
  reported fit is the 2026-09-25 run of this script). Writes
  `results/hierarchical_fit.pkl` with the full joint posterior.
- `slurm_param_recovery.sh <mode>` — array job for `08_param_recovery.py`:
  task `k` refits synthetic dataset `k` of `<mode>` (`prior`, `null` or
  `posterior`). Generate the datasets first with
  `python 08_param_recovery.py simulate --mode <mode> --n_rep 8` (needs the
  real fit), and summarise afterwards with
  `python 08_param_recovery.py analyze --mode <mode>`. A re-submitted task
  resumes from its checkpoints.

Everything after the fit (`02_postprocess.py` onwards) runs in minutes on a
laptop: copy `results/hierarchical_fit.pkl` back and follow
`analyses/hmm/README.md`.

## Configuration

The fit reads `N_OUTER`, `BURN`, `THIN`, `N_CHAINS`, `N_FFBS`, `CKPT_EVERY`,
`RESUME` from the environment; the defaults in `jointlearn/hmm/sampler.py` are
the reported configuration and the fit script uses the same values.

On Snellius the caller's environment is **not** exported to the job, so
`N_CHAINS=4 sbatch ...` is silently ignored. Pass overrides through sbatch:

```bash
sbatch --export=ALL,N_CHAINS=4 slurm_fit_hierarchical.sh
```

Smoke test that does not touch the production `results/`:

```bash
sbatch --time=00:30:00 \
  --export=ALL,EA_RESULTS_DIR=$HOME/ea_smoke/results,EA_FIGURES_DIR=$HOME/ea_smoke/figures,N_OUTER=20,BURN=5,THIN=1,N_CHAINS=2 \
  slurm_fit_hierarchical.sh
```

## Prerequisites

- The repository checked out on Snellius; the data file
  `data/langlearning_v2_anonymized.csv` is part of it.
- Conda env `pymc523` with CUDA-enabled JAX
  (`python -c "import jax; print(jax.devices())"` should list a GPU; otherwise
  `pip install -U "jax[cuda12]"`).

## Notes

- `--partition=gpu_a100` is the Snellius A100 partition (`gpu_h100` for
  H100; check current names with `sinfo -s`). The code uses one GPU.
- `OMP_NUM_THREADS=1` etc. avoid CPU oversubscription;
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`
  stop JAX from grabbing 90% of the GPU up front.
- The fit's 3 h time limit is a safety margin; if a job is killed anyway,
  resubmit with `--export=ALL,RESUME=1` and it continues from its checkpoints.
