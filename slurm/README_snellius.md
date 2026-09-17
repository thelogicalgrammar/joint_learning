# Running the JAX pipeline on Snellius

GPU-backed full-scale run of `01_fit.py` + `02_postprocess.py`. Conda env `pymc523`, threading caps and SLURM directives as in the other scripts here.

## Files

- `slurm_sampler.sh` — hierarchical fit on 1 GPU (gpu partition). ~25 min wall on a laptop RTX 3050 Ti with the per-word belief FFBS (0.12 s per outer iteration x 1200 x 14 chains); the 18 h time limit is a large safety margin. Saves the full joint posterior including `(B, O)` for all 325 participants.
- `slurm_02_postprocess.sh` — post-process the hierarchical fit into per-participant marginals + PPC. **No MCMC**, just summarises what's in `hierarchical_fit.pkl`. Runs in seconds. (Kept on GPU partition only to share the JAX env; no GPU needed.)
- `slurm_param_recovery.sh` — parameter recovery for the hierarchical fit as an array job: task `k` refits synthetic dataset `k` of `MODE` (`prior`, `null` or `posterior`; see `analyses/hmm/08_param_recovery.py`). Generate the datasets first with `python 08_param_recovery.py simulate --mode $MODE --n_rep 8` (needs `results/hierarchical_fit.pkl`), then `MODE=posterior sbatch slurm_param_recovery.sh`, then `python 08_param_recovery.py analyze --mode $MODE`. Each task takes a few minutes at the recovery defaults (`N_OUTER=1200 BURN=400 THIN=8 N_CHAINS=3`); a re-submitted task resumes from its checkpoints.

The hierarchical fit reads its config from env vars (`N_OUTER`, `N_CHAINS`, etc.) with the local-run defaults built in. The post-process script has no config of its own.

## Config (env vars exported by the SLURM scripts)

|                         | local default       | Snellius (full) |
|-------------------------|---------------------|-----------------|
| **hierarchical fit**    |                     |                 |
| `N_OUTER`               | 600                 | 1200            |
| `BURN`                  | 200                 | 400             |
| `THIN`                  | 4                   | 4               |
| `N_CHAINS`              | 7                   | 14              |
| `N_FFBS`                | 1                   | 1               |
| `N_SWAP`                | 0                   | 0               |
| **per-ppt post-process** | (no MCMC, just summarises) | (no MCMC, just summarises) |

## Workflow

From the project root on Snellius:

```bash
cd slurm
sbatch slurm_sampler.sh
# wait for completion (or watch with `squeue -u $USER` / `tail -f hier_fit_<jobid>.out`)
sbatch slurm_02_postprocess.sh
```

The inference script auto-checks for `results/hierarchical_fit.pkl` and exits early if it's missing — so you can submit both jobs in sequence and the second will fail-fast if the first hasn't finished.

Local downstream plotting / diagnostics (no GPU needed) on the resulting pickles:

```bash
cd ../analyses/hmm
python 03_condition_effects.py      # main result: condition-level posteriors
python 04_convergence.py
python 04_convergence.py
python 06_per_participant.py
python 05_ppc.py
python 05_ppc.py                    # posterior predictive p-values
python 07_survival.py                # secondary survival cross-checks
python 07_survival.py
```

## Prerequisites on Snellius

1. **Repo present** under a stable path (e.g. `~/joint_learning`). The SLURM scripts use relative paths (`cd ../analyses/hmm`) so submit from `slurm/`.
2. **Data file** `data/langlearning_v2_anonymized.csv` is part of the repository (path set by `DATA_PATH` in `analyses/hmm/data.py`), so nothing needs to be copied.
3. **Conda env `pymc523`** with JAX-CUDA installed:
   ```bash
   conda activate pymc523
   python -c "import jax; print(jax.devices())"   # should print [cuda(id=0)] or similar
   ```
   If JAX is CPU-only on the cluster, install with:
   ```bash
   pip install -U "jax[cuda12]"
   ```
   (matching the Snellius CUDA module; check with `module avail cuda`).
4. **`lifelines`, `arviz`, `xarray`** for the downstream analyses (already pulled in by the local pipeline).

## SLURM directives reference

- `--partition=gpu` — Snellius A100 GPU partition. For H100 use `--partition=gpu_h100`. For MIG-sliced shares use `--partition=gpu_mig`.
- `--gpus-per-node=1` — one GPU per job. The JAX code is single-GPU; more would sit idle.
- `--cpus-per-task=16` — generous because we set `OMP_NUM_THREADS=1` to avoid oversubscription; spare cores help Python I/O and `numpy` accumulation during sampling.
- `--time=18:00:00` (fit) / `12:00:00` (infer) — generous limits left over from the block-move sampler; the fit now takes well under an hour. Snellius `gpu` partition allows up to 5 days.

## Verifying setup before the big run

Submit a tiny smoke-test with reduced sample counts to confirm the env loads and the fit runs at all:

```bash
EA_RESULTS_DIR=$HOME/ea_smoke/results EA_FIGURES_DIR=$HOME/ea_smoke/figures \
N_OUTER=20 BURN=5 THIN=1 N_CHAINS=2 \
sbatch --time=00:30:00 slurm_sampler.sh
```

Should complete in a few minutes and write `$HOME/ea_smoke/results/hierarchical_fit.pkl`. The wrapper only fills in `N_OUTER`, `BURN`, `THIN`, `N_CHAINS`, `N_FFBS`, `N_SWAP`, `CKPT_EVERY`, `RESUME` when they are not already set, so values given on the `sbatch` line win. Set `EA_RESULTS_DIR` as above so the smoke test does not overwrite the production `results/` pickles and checkpoints. Don't rely on the smoke-test fit for actual analysis.

## Notes

- The SLURM scripts set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. The default JAX behaviour of pre-allocating 90% of GPU memory often fights with the SLURM scheduler's MIG/cgroup limits; opting out and capping at 85% is safer on shared GPUs.
- The `_count_moves` / `_movetype` swap-rule patch (at-least-one-heard, both consistent) is already in `model.py`. The Snellius run uses the fixed rule.
- The PPC in `02_postprocess.py` draws one replicate per joint `(B, O, tpar)` posterior sample (a mixed posterior predictive check, conditional on the inferred latent trajectories). It needs a fit pickle with full `B`/`O` arrays, i.e. one produced by the current `01_fit.py`.
