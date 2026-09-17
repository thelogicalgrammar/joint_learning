# Server jobs

SLURM scripts for the project's compute-heavy jobs.

## Files

- `slurm_sampler.sh` — GPU-partition hierarchical fit for the JAX HMM (`model.analyses/hmm.sampler`). See `README_snellius.md`.
- `slurm_02_postprocess.sh` — fast post-process step that derives per-ppt marginals + PPC from the hierarchical fit pickle. See `README_snellius.md`.
- `slurm_param_recovery.sh` — array job for the HMM parameter recovery (`analyses/hmm/08_param_recovery.py`). See `README_snellius.md`.
- `README_snellius.md` — workflow + config table for the JAX-pipeline jobs on Snellius.
