# Learning what comes first

Analyses of a 200-trial artificial-language learning experiment (Aclapa
task: 4-alternative picture selection, one of the six word orders per
participant, 386 L1-English participants) asking whether word-order
learnability mirrors typological frequency.

## Layout

- `data/` — the anonymized behavioural dataset shipped with the repository
  and the script that derives it from the raw export (see `data/README.md`).
- `model/scripts/` — shared loaders: `data_functions.py` (reads the data,
  applies the participant exclusion, builds the trial-level arrays),
  `simulation_functions.py` (the stimulus space: scenes, languages, word
  orders; `define_objects`), `model_free_analysis.py` (logistic regression
  of accuracy on trial and condition with `bambi`).
- `model/bayesian.ipynb` — simulation of an idealized Bayesian learner of
  the artificial language (hypothesis elimination over lexicon and word
  order, with an attention ceiling), the motivation for the learning-curve
  shape; figures in `model/figures/`.
- `model/logistic_model/` — the model-free analysis in R (`script.R`:
  mixed-effects logistic regression of accuracy over trials by condition,
  with the participants' language background), plus its two figures.
- `model/experiment_analysis/` — the hierarchical partial-lexicon HMM
  (JAX): a measurement model that infers, per participant and trial, which
  words are known and whether a word order has been committed to, with a
  condition-level hyperprior on the learning parameters. Includes fitting,
  diagnostics, posterior predictive checks and parameter recovery. See its
  `README.md`.
- `server_jobs/` — SLURM scripts for running the HMM fit, its
  post-processing and the parameter recovery on a GPU cluster
  (`README_snellius.md`).

## Environment

Python ≥ 3.11 with the packages in `requirements.txt` (JAX with CUDA
support for the HMM fit; CPU works for smoke tests). The R script needs
`tidyverse`, `lme4`, `emmeans`, `tidybayes`.

## Reproducing the HMM analysis

```bash
cd model/experiment_analysis
python fit_hierarchical.py            # ~1 h on a laptop GPU at the production config
python infer_per_participant.py
python analyze_condition_effects.py   # main result
python ppc_pvalues.py                 # posterior predictive checks
```

The full list of scripts, outputs and the production configuration is in
`model/experiment_analysis/README.md`.
