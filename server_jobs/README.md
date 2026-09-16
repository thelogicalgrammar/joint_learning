# Server jobs

SLURM scripts for the project's compute-heavy jobs.

## Files

- `slurm_fit_hierarchical.sh` — GPU-partition hierarchical fit for the JAX HMM (`model.experiment_analysis.fit_hierarchical`). See `README_snellius.md`.
- `slurm_infer_per_participant.sh` — fast post-process step that derives per-ppt marginals + PPC from the hierarchical fit pickle. See `README_snellius.md`.
- `slurm_param_recovery.sh` — array job for the HMM parameter recovery (`model/experiment_analysis/param_recovery.py`). See `README_snellius.md`.
- `README_snellius.md` — workflow + config table for the JAX-pipeline jobs on Snellius.
