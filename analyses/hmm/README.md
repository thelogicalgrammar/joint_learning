# Lexicon + Word-Order Bayesian Model

Analyses of the Aclapa 200-trial 4AFC word-learning experiment, fitting a
partial-lexicon HMM with an explicit UNK / committed-order distinction.
JAX implementation, GPU-friendly. Hierarchical Bayesian HMC-within-Gibbs sampler with a
**condition-level hyperprior** (condition = the participant's word order),
so that differences in learnability between word orders are estimated inside
the model rather than by post-hoc tests on latent quantities. All
per-participant (B, O, tpar) joint samples are persisted, so every
downstream analysis is derived deterministically from a single fit pickle.

## Model in one paragraph

Per participant and trial, the latent state is a partial lexicon `B[t, word]
∈ {UNK, meaning 0..6}` and an order state `O[t] ∈ {SVO, …, OVS, UNK}`.
Beliefs change by at most one move per trial (single-word change with
log-cost κ, two-word swap with log-cost κ_s); the order commits from UNK at
per-trial rate γ_o (= `p_commit_o`) and switches between committed orders at
rate ε_o. Choices are uniform over the candidate scenes compatible with the
current state (role-position match under a committed order, bag-of-words
match under UNK), mixed with a lapse rate λ. Four per-participant parameters
are fitted on transformed scales, `(logit λ, log κ, logit ε_o, logit γ_o)`,
with the hierarchy

```
tpar[p, k] ~ N(mu[w(p), k], sigma_k^2)        participant within condition
mu[w, k]   ~ N(m_k, tau_k^2)                  condition mean (6 word orders)
m_k        ~ N(mu0_k, sd0_k^2)                population mean (config.py)
sigma_k^2  ~ InvGamma(A0, B0),   tau_k^2 ~ InvGamma(A_TAU, B_TAU)
```

All continuous parameters are sampled jointly by HMC (non-centred). `tau_k` measures how much the six
condition means differ; the posterior of `mu[:, k]` gives the condition
contrasts (see `03_condition_effects.py`).

### Swap cost κ_s is fixed, not fitted

`model.KAPPA_S_FIXED = 20`. In every posterior sample of every fit so far the
belief trajectories contained **zero** two-word swap transitions. With no
swaps in the sampled trajectories the likelihood is monotone increasing in
κ_s (it enters only through the per-trial normaliser Z), so a fitted κ_s
simply drifts upward until the hyperprior stops it: in the earlier fits its
population value rose within every chain and never converged. κ_s is
therefore unidentified from these data. The swap *transition type* stays
in the model (the reported fit was run with it); at κ_s = 20 such a
transition is ~e⁻²⁰ as likely as staying put, i.e. negligible. If a future
dataset does show swaps, re-introduce κ_s as a fifth fitted parameter in
`model.to_nat` / `config.py`.

## Layout

The model is a library, `jointlearn/hmm/` (importable after `pip install -e .`
from the repository root); the analysis is a pipeline of numbered scripts in
this folder that are run in order, each reading the previous step's output
from `results/` (override the folders with the env vars `EA_RESULTS_DIR`,
`EA_FIGURES_DIR`, e.g. for a smoke run or the parameter recovery). Tests of
the kernels are in `../../tests/`.

```
jointlearn/hmm/                    library
  model.py  config.py  dataset.py  sampler.py  simulate.py  io.py  analysis.py
analyses/hmm/                      pipeline (this folder)
  01_fit.py               ──► results/hierarchical_fit.pkl  (joint posterior: B, O, tpar for all
                              ppts; hyper = {mu, sigma, m, tau}; all chains) + hier_chain{k}.pkl
  02_postprocess.py       ──► results/per_participant_{marginals,ppc}.pkl, hierarchical_fit.sig.json
  03_condition_effects.py ──► figures/{condition_effects,commitment_curves}.pdf, results/03_condition_effects.md  (paper: condition effects)
  04_convergence.py       ──► hierarchical_fit_summary.png, hyperprior_predictive.png
  05_ppc.py               ──► figures/ppc.pdf, results/05_ppc.md       (paper: posterior predictive checks)
  06_per_participant.py   ──► figures/participants.pdf (paper), per_participant.pdf (supplement), results/06_per_participant.md
  08_param_recovery.py    ──► results/param_recovery/, param_recovery_<mode>.png
tests/test_ffbs.py, tests/test_simulate.py
```

## Library (`jointlearn/hmm/`)

- **`model.py`** — partial-lexicon HMM with UNK-order extension, in JAX. Pure-functional, jit-able, vmap-able. Emission likelihood, belief transitions, order transition matrix, `traj_sweep` (one Gibbs pass = FFBS over order + exact per-word FFBS over belief (`ffbs_word` / `ffbs_beliefs`)), `loglik`, sufficient statistics, and `to_nat` (the single source of truth for the transformed→natural parameter mapping, including the fixed κ_s).
- **`config.py`** — the hierarchy's prior and initialisation constants (documented in the file).
- **`dataset.py`** — loads experimental data (signals, candidate scenes, choices, true word-order assignments = condition, accuracy) via `jointlearn.data.get_data`. Imported as `RP` by the pipeline scripts. With `EA_SIM_DATA=<file.npz>` the observed choices are replaced by simulated ones on the same stimuli (parameter recovery, see `08_param_recovery.py`); everything else is unchanged.
- **`sampler.py`** — the HMC-within-Gibbs sampler: batched JAX arrays of the data, the vmap'd (B, O) sweep, the sufficient statistics collapsed into counts (`hist_stats`, `loglik_hist`), the NumPyro model (non-centred hierarchy), `JitterHMC`, `start_values`, `run_chain`, `save_fit`. Configured by env vars at import (`N_WARMUP, N_SAMPLES, THIN, N_CHAINS, TRAJ, SEED0, RESUME`). `01_fit.py` is its entry point.
- **`simulate.py`** — forward simulation from the model: `simulate_latents` draws (B, O) from the transition prior for one participant, `simulate_choices` / `choice_probs` draw choices from the emission model given trajectories (the same mixture `emission_lp` evaluates; used by `02_postprocess.py` for the PPC replicates too), `save_sim_dataset` writes an `EA_SIM_DATA` file. Note the model is a measurement model: prior-simulated lexicons are random walks that do not converge to the truth, so accuracy against the true scene stays near chance in a prior simulation.
- **`io.py`** — `RESULTS_DIR` / `FIGURES_DIR`, pickle IO, `load_fit` (refuses pickles whose participant/condition vector, `n_tpar` or fixed kappa_s do not match the current data and code), the fit signature / `load_derived` staleness guard, chain pooling, `savefig`.
- **`analysis.py`** — `bag_consistent_mask` / `compute_swap_mask` (candidates consistent with the utterance up to word order; the word-order-discriminating trials, ≈9%), `pack_choices` / `unpack_choices` (2-bit packing of 4AFC choice sequences) and the `violins` plot helper.

## Pipeline (this folder)

- **`01_fit.py`** — hierarchical HMC-within-Gibbs (NumPyro `HMCGibbs`) over all participants on GPU. Per iteration: (1) Gibbs step: every participant's (B, O) trajectories redrawn exactly by `model.traj_sweep` (FFBS) given its parameters, vmapped over participants, and summarised into 40 sufficient counts per participant (the likelihood depends on the per-trial statistics only through how often each value occurs; the swap term, negligible at κ_s = 20, is dropped); (2) HMC step: all continuous parameters (m, τ², σ², condition means, participant parameters; non-centred, `mu = m + tau * mu_raw`, `tpar = mu[w] + sigma * z`) jointly, with a fixed-length trajectory whose length is redrawn every iteration uniformly in [0.5, 1.5] × `TRAJ` (avoids periodicity; step size and diagonal mass matrix adapted in warm-up). NUTS samples the same posterior but its tree building costs ~8× the gradient here. Chains differ by seed (`SEED0 + k`) and a dispersed start; order-specific inits are impossible because FFBS resamples O at every sweep. Each chain runs in its own process (several chains in one process exhaust a 4 GB GPU) and writes `results/hier_chain{k}.pkl`; the script then assembles `results/hierarchical_fit.pkl`. `RESUME=1` reuses finished chains with the same config; `CHAIN=k python 01_fit.py` runs one chain only. Draws are not bit-reproducible across runs on a GPU (non-deterministic float reductions, amplified by HMC), only distributionally. **Defaults = the reported fit (2026-11-10, laptop RTX 3050 Ti): 8 chains × (1000 warm-up + 6000 iterations, thin 20) = 2400 pooled draws, `TRAJ=3.3`, `SEED0=20000`; ~15 min per chain (~0.1 s per iteration, dominated by the Gibbs sweep), ~2 h in all; 0 divergences. All condition means R-hat ≤ 1.019 and ESS 473–2457 (logit_eps_o 1700–2110; logit_p_commit_o, the parameter carrying the result, ≥ 473); per-participant logit(λ) R-hat > 1.1 for 2/325. The previous reported fit (hand-written Gibbs/MH sampler with group shift moves, 14 chains, 2026-09-25; logit_eps_o ESS only 179–420) gives the same results within Monte Carlo error; it and its outputs are archived in `results/archive_mh_14chain_2026-09-25/`, the sampler in git history (before the HMC switch).**
- **`02_postprocess.py`** — *post-process only*, no MCMC. Reads the joint posterior, derives per-participant marginals and a mixed posterior predictive check (one replicate choice sequence per joint (B, O, tpar) sample, conditional on the inferred latent trajectories). Requires a fit pickle produced by the current `01_fit.py`. Writes:
  - `hierarchical_fit.sig.json` — signature of the fit (seeds, config, hash of the tpar samples). Every pickle below embeds it, and the consumer scripts load them via `io.load_derived`, which refuses a pickle derived from a different fit than the current `hierarchical_fit.pkl` (i.e. the fit was re-run but this script was not).
  - `per_participant_marginals.pkl` — averaged belief_marg + order_marg + posterior-mean natural-scale params `(λ, ε_o, κ, κ_s, γ_o)` per ppt.
  - `per_participant_ppc.pkl` — `pred_p_wrong`, per-ppt Bayesian p-values (`p_total`, `p_swap`), error-count replicates, observed counts, swap mask; per joint sample also the replicate choice sequence (`rep_choice_packed`, 2 bits per trial) and the emission log-likelihood of the observed and of the replicated choices (`ll_obs`, `ll_rep`), from which `05_ppc.py` computes posterior predictive p-values for arbitrary test statistics.
- **`03_condition_effects.py`** — the paper's "Condition effects" section, in the paper's order: figure (γ, κ, λ per word order), figure and table of the share of participants committed to the *correct* order by trial and the trial by which half have (from the sampled latent trajectories; γ itself is the rate of committing to *any* order, the order being chosen by the data), table of posteriors per word order, typological-group ratios and the ranking for the text, and the table of all pairwise comparisons. Values are for the typical participant of a condition (inverse link of `mu[j, k]`). Writes `figures/condition_effects.pdf` and the report `results/03_condition_effects.md` (Markdown tables plus LaTeX to paste into the paper).
- **`04_convergence.py`** — `convergence()`: R-hat across chains on per-participant logit(λ), on the condition means and on the population means; HMC diagnostics per chain (leapfrog steps, step size, divergences); population-λ trace (`hierarchical_fit_summary.png`). `hyperprior_predictive()`: new-participant prior vs posterior predictive per parameter on the natural scale (conditions pooled), integrating over the full hierarchy on both sides, with practical interpretations annotated (`hyperprior_predictive.png`).
- **`05_ppc.py`** — the paper's "Posterior predictive checks" section, in the paper's order: table of pooled test statistics (choice log-likelihood, error rate, error clustering, share of role-swap errors on the swapped picture; observed vs replicated, posterior predictive p), per-participant log-likelihood p-values and block counts for the text, the figure (errors per 25-trial block; errors by number of tied candidates), and the table of errors by tied candidates. A *mixed* PPC (one replicate per joint (B, O, tpar) draw, conditional on the inferred latent trajectories), hence conservative: it tests the choice model and the error structure, not the transition prior. A fully marginal PPC is deliberately not done: the HMM is a measurement model whose prior does not pull beliefs toward the true lexicon. Writes `figures/ppc.pdf` and the report `results/05_ppc.md`.
- **`06_per_participant.py`** — one compact panel per participant: stacked posterior of the word-order state (correct / wrong / UNK), the same per word, and the posterior predictive P(error) with observed errors and role-swap trials. Paper figure `participants.pdf` with four example participants (fast SVO, gradual SOV, wrong-order commitment, OSV learner who never commits; IDs and rationale in the docstring, numbers in `results/06_per_participant.md` with the LaTeX); supplement `per_participant.pdf` with every participant, six per page, by condition then accuracy.
- **`08_param_recovery.py`** — `simulate` / `fit` / `analyze` for synthetic datasets that keep the real stimuli, participants and conditions and replace the choices (`--mode prior`: truth = posterior means of the real fit's hyperparameters, per-participant tpar and latent trajectories drawn from the model, a weak test because a random-walk lexicon contradicts all four candidates on most trials, so ~77% of the simulated trials have a uniform emission vs ~19% in `posterior` mode; `--mode null`: the same with all six condition means equal, a false-positive check for the condition contrasts; `--mode posterior`: one joint posterior draw of the real fit is the truth and only the choices are regenerated, i.e. learning-like trajectories, the regime of the real data). `fit` runs `01_fit.py` in a subprocess per dataset (`EA_SIM_DATA`, `EA_RESULTS_DIR=results/param_recovery/<mode>/rep<k>/`; recovery defaults `N_WARMUP=500 N_SAMPLES=2000 THIN=20 N_CHAINS=3 RESUME=1`, overridable through the same env vars). `analyze` reports 95% coverage, bias, RMSE and correlation of the condition means and per-participant tpar, coverage of the hyper sds, the typological p_commit contrasts (true vs recovered, and the false-positive count in `null` mode) and latent recovery (posterior probability of the true order / belief per trial, commit-time error), and writes `figures/param_recovery_<mode>.{png,pdf}` (condition means and per-participant parameters, true vs estimated). `../../slurm/slurm_param_recovery.sh` fans the fits out as a SLURM array job.

## Tests (`tests/`)

- **`test_ffbs.py`** — correctness checks for the per-word belief FFBS: (1) the analytic per-word transition table equals `_transition_lp` on all 64 candidate pairs, entry for entry, on sampled and randomly perturbed beliefs; (2) brute-force exactness: on 7-trial windows of real data all 8^6 trajectories of one word are enumerated and scored with the generic model functions, and the exact conditional is compared with the empirical distribution of `ffbs_word` draws (total variation at the Monte Carlo floor, zero mass on infeasible states). Run after any change to the belief kernel. ~2–3 min on GPU.
- **`test_simulate.py`** — checks `simulate.py` against `model.py`: simulated transitions are feasible and their frequencies match the transition probabilities; `choice_probs` equals `exp(emission_lp)`; the pooled profile log-likelihood of simulated participants peaks at the true parameters. ~1 min on GPU.

Both run as plain scripts (`python tests/test_ffbs.py`, prints PASS / FAIL) or under `pytest`.

## Output

- `figures/condition_effects.pdf` — paper figure: posterior per word order (main result); `results/03_condition_effects.md` — the section's numbers and LaTeX
- `figures/hierarchical_fit_summary.png` — multi-chain diagnostic
- `figures/hyperprior_predictive.png` — prior vs posterior per param
- `figures/participants.pdf` — paper figure: four example participants; `figures/per_participant.pdf` — supplement: all participants, six per page
- `figures/ppc.pdf` — paper figure: posterior predictive checks; `results/05_ppc.md` — the section's numbers and LaTeX
- `figures/param_recovery_<mode>.png` — parameter recovery (`prior`, `null`, `posterior`)

## Dependencies

- `numpy`, `scipy`, `matplotlib`, `pandas` — standard
- `jax` (with CUDA support for GPU use) — pure-functional model + Gibbs inner loop
- `numpyro` — HMC-within-Gibbs (`HMCGibbs`)
- `arviz` (+ `xarray`) — R-hat diagnostics

The conda env `pymc523` has all of these.

## To reproduce from scratch

From the repository root, once: `pip install -e .` (makes `jointlearn` importable). Then, in this folder:

```bash
python 01_fit.py                  # ~2 h on a laptop GPU at the defaults (= reported fit, 8 chains of ~15 min)
python 02_postprocess.py          # minutes (post-process, no MCMC)
python 03_condition_effects.py    # main result
python 04_convergence.py
python 05_ppc.py
python 06_per_participant.py
```

Parameter recovery (needs the real fit for the ground truth; ~15 min per dataset on a laptop GPU at the recovery defaults):

```bash
python 08_param_recovery.py simulate --mode posterior --n_rep 8   # also: prior, null
python 08_param_recovery.py fit --mode posterior                  # or one dataset: --rep 3
python 08_param_recovery.py analyze --mode posterior
```

Quick end-to-end smoke test on CPU without touching `results/` or `figures/`:

```bash
export EA_RESULTS_DIR=/tmp/ea_smoke/results EA_FIGURES_DIR=/tmp/ea_smoke/figures
JAX_PLATFORMS=cpu N_WARMUP=10 N_SAMPLES=8 THIN=2 N_CHAINS=2 python 01_fit.py
python 02_postprocess.py && python 03_condition_effects.py   # etc.
```

For Snellius (larger config, GPU partition), see `../../slurm/README.md`.
