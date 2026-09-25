# Learning what comes first

Analyses of a 200-trial artificial-language learning experiment (Aclapa
task: 4-alternative picture selection, one of the six word orders per
participant, 386 L1-English participants) asking whether word-order
learnability mirrors typological frequency.

## Layout

- `data/` — the anonymized behavioural dataset shipped with the repository
  and the script that derives it from the raw export (see `data/README.md`).
- `jointlearn/` — the shared library (`pip install -e .`): `data.py` reads
  the data, applies the participant exclusion and builds the trial-level
  arrays; `simulation.py` defines the stimulus space (scenes, languages, word
  orders) and simulates idealized Bayesian learners; `hmm/` is the
  hierarchical partial-lexicon HMM (model, sampler, simulator, IO).
- `analyses/ideal_learner/` — `bayesian.ipynb`: simulation of an idealized
  Bayesian learner of the artificial language (hypothesis elimination over
  lexicon and word order, with an attention ceiling), the motivation for the
  learning-curve shape; figures alongside.
- `analyses/model_free/` — the model-free analysis: `script.R` (mixed-effects
  logistic regression of accuracy over trials by condition, with the
  participants' language background) and its Python counterpart
  `model_free_analysis.py` (bambi), plus two figures.
- `analyses/hmm/` — the HMM pipeline as numbered steps (fit, post-process,
  condition effects = main result, convergence, posterior predictive checks,
  per-participant plots, survival cross-checks, parameter recovery). See its
  `README.md`.
- `tests/` — correctness checks of the HMM kernels and simulator (plain
  scripts, also collected by `pytest`).
- `slurm/` — SLURM scripts for the HMM fit, its post-processing and the
  parameter recovery on a GPU cluster (`README_snellius.md`).

## Environment

Python ≥ 3.11 with the packages in `requirements.txt` (JAX with CUDA
support for the HMM fit; CPU works for smoke tests), then `pip install -e .`
from this directory so that `jointlearn` is importable from anywhere. The R
script needs `tidyverse`, `lme4`, `emmeans`, `tidybayes`.

## Reproducing the HMM analysis

```bash
cd analyses/hmm
python 01_fit.py                  # ~1 h on a laptop GPU (defaults = the reported configuration)
python 02_postprocess.py
python 03_condition_effects.py    # main result
python 05_ppc.py                  # posterior predictive checks
```

The full list of steps, outputs and the production configuration is in
`analyses/hmm/README.md`.
