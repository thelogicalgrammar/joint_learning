# Lexicon + Word-Order Bayesian Model

Analyses of the Aclapa 200-trial 4AFC word-learning experiment, fitting a
partial-lexicon HMM with an explicit UNK / committed-order distinction.
JAX implementation, GPU-friendly. Hierarchical Bayesian Gibbs sampler with a
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

All hyper-updates are conjugate. `tau_k` measures how much the six
condition means differ; the posterior of `mu[:, k]` gives the condition
contrasts (see `03_condition_effects.py`).

### Swap cost κ_s is fixed, not fitted

`model.KAPPA_S_FIXED = 20`. In every posterior sample of every fit so far the
belief trajectories contained **zero** two-word swap transitions. With no
swaps in the sampled trajectories the likelihood is monotone increasing in
κ_s (it enters only through the per-trial normaliser Z), so a fitted κ_s
simply drifts upward until the hyperprior stops it: in the earlier fits its
population value rose within every chain and never converged. κ_s is
therefore unidentified from these data. The swap *transition type* is kept
only so that `swap_move` (an MH proposal that relabels two words from a
trial onward, which lets the sampler escape lexicon-permutation modes)
remains admissible; at κ_s = 20 such a transition is ~e⁻²⁰ as likely as
staying put, i.e. accepted only on overwhelming evidence. If a future
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
                              ppts; hyper = {mu, sigma, m, tau}; all chains) + checkpoints
  02_postprocess.py       ──► results/per_participant_{settle,marginals,ppc}.pkl, hierarchical_fit.sig.json
  03_condition_effects.py ──► figures/condition_effects.png            (main result)
  04_convergence.py       ──► hierarchical_fit_summary.png, hyperprior_predictive.png
  05_ppc.py               ──► ppc_*.png                                (posterior predictive checks)
  06_per_participant.py   ──► per_participant.pdf
  07_survival.py          ──► logrank.png, weibull.png                 (secondary)
  08_param_recovery.py    ──► results/param_recovery/, param_recovery_<mode>.png
tests/test_ffbs.py, tests/test_simulate.py
```

## Library (`jointlearn/hmm/`)

- **`model.py`** — partial-lexicon HMM with UNK-order extension, in JAX. Pure-functional, jit-able, vmap-able. Emission likelihood, belief transitions, order transition matrix, `traj_sweep` (one Gibbs pass = FFBS over order + exact per-word FFBS over belief (`ffbs_word` / `ffbs_beliefs`) + swap MH; the old block-move MH is still available via `n_block` but off by default: on real data its blind proposals were accepted ~0.3% of the time and cost ~70% of the runtime), `loglik`, `emit_p_correct`, sufficient statistics, and `to_nat` (the single source of truth for the transformed→natural parameter mapping, including the fixed κ_s).
- **`config.py`** — the hierarchy's prior and initialisation constants (documented in the file).
- **`dataset.py`** — loads experimental data (signals, candidate scenes, choices, true word-order assignments = condition, accuracy) via `jointlearn.data.get_data`. Imported as `RP` by the pipeline scripts. With `EA_SIM_DATA=<file.npz>` the observed choices are replaced by simulated ones on the same stimuli (parameter recovery, see `08_param_recovery.py`); everything else is unchanged.
- **`sampler.py`** — the hierarchical Gibbs sampler: batched JAX arrays of the data, the vmap'd sweep / sufficient-statistics / MH / group-shift kernels, the conjugate hyper update, `run_chain` (with checkpointing) and `save_fit`. Configured by env vars at import (`N_OUTER, BURN, THIN, N_CHAINS, N_FFBS, N_SWAP, N_BLOCK, CKPT_EVERY, RESUME`). `01_fit.py` is its entry point; the algorithm is described there.
- **`simulate.py`** — forward simulation from the model: `simulate_latents` draws (B, O) from the transition prior for one participant, `simulate_choices` / `choice_probs` draw choices from the emission model given trajectories (the same mixture `emission_lp` evaluates; used by `02_postprocess.py` for the PPC replicates too), `save_sim_dataset` writes an `EA_SIM_DATA` file. Note the model is a measurement model: prior-simulated lexicons are random walks that do not converge to the truth, so accuracy against the true scene stays near chance in a prior simulation.
- **`io.py`** — `RESULTS_DIR` / `FIGURES_DIR`, pickle IO, `load_fit` (refuses pickles from older model versions, and pickles whose participant/condition vector, `n_tpar` or fixed kappa_s do not match the current data and code), the fit signature / `load_derived` staleness guard, chain pooling, `savefig`.
- **`analysis.py`** — `bag_consistent_mask` / `compute_swap_mask` (candidates consistent with the utterance up to word order; the word-order-discriminating trials, ≈9%), `pack_choices` / `unpack_choices` (2-bit packing of 4AFC choice sequences), `settle_matrix` (first-passage times of sampled O to the true order), `km_frame` and the `violins` plot helper.

## Pipeline (this folder)

- **`01_fit.py`** — hierarchical Gibbs over all participants on GPU. JAX vmap across participants; Python loop across `N_CHAINS` chains that differ only by PRNG seed (order-specific inits are impossible: FFBS resamples O at the start of every sweep, so convergence diagnostics rest on multi-seed agreement). Per outer iteration: (B, O) sweep → sufficient stats → coordinate-wise MH on tpar per participant given its condition mean (one random-walk proposal per transformed coordinate with its own scale, `config.PROP_SD`; the earlier joint isotropic proposal was vetoed by the tight log_kappa coordinate and crawled on logit_p_commit) → group shift moves (`shift_all`: for each coordinate, one common delta added to a condition mean and all its participants, accepted per condition; then one common delta added to the population mean, all condition means and all participants. Under a common shift the within-condition and condition-level Gaussian terms cancel, so only the likelihood change and the top-level prior enter. This breaks the slow alternation between data-starved participants and their condition mean, which capped the eps_o condition-mean ESS at a few draws per hundred; scales in `config.SHIFT_SD` / `SHIFT_SD_POP`, tuned for ~0.35–0.5 acceptance) → conjugate updates of `mu`, `sigma`, `m`, `tau`. Logs the per-coordinate acceptance rates of all three moves (targets ~0.44 for coordinate-wise MH, ~0.35–0.5 for the shifts). Saves the full joint posterior and the prior config. Config via env vars `N_OUTER, BURN, THIN, N_CHAINS, N_FFBS, CKPT_EVERY` (plus `N_SWAP` and `N_BLOCK` for the swap-MH and legacy block-move kernels, both default 0: with per-word FFBS neither changes the sampled marginals, and they cost ~40% and ~600% of the sweep respectively). Checkpoints (`results/hier_chain{k}_ckpt.pkl`, every `CKPT_EVERY` iterations and at the end of each chain) carry the full sampler state; with `RESUME=1` finished chains are reused and an interrupted chain continues exactly where it stopped (a finished chain can also be extended by re-running with a larger `N_OUTER`). A checkpoint is only resumed if its config tuple matches the current code: kernel settings, proposal scales, all prior constants in `config.py`, `model.KAPPA_S_FIXED`, and the participant/condition vector; otherwise the chain restarts. A checkpoint that is further along than the requested `N_OUTER` is refused with an error (re-run with a larger `N_OUTER` or `RESUME=0`). **Defaults = the reported fit: 4800 outer iterations, burn-in 1200, thin 12 (300 retained samples per chain), 14 chains (4200 pooled draws; run on a Snellius A100 on 2026-09-25, `slurm/slurm_fit_hierarchical.sh`); ~0.08 s per outer iteration on a laptop RTX 3050 Ti, i.e. ~110 min there, ~55 min for 7 chains (the old block-move sampler took 2.5 s per iteration). With that configuration all condition means had R-hat ≤ 1.05 and ESS 179–4368 (the lowest values are all on logit_eps_o, the weakly identified order-switch rate; on logit_p_commit_o, the parameter carrying the result, R-hat ≤ 1.024 and ESS ≥ 517), and the condition-level posteriors agree with the earlier 7-chain laptop fit within Monte Carlo error. Shorter runs (e.g. `N_OUTER=600 BURN=200 THIN=4`, ~6 min) are fine for a quick look but the burn-in is then only a few autocorrelation times of the p_commit condition means.**
- **`02_postprocess.py`** — *post-process only*, no MCMC. Reads the joint posterior, derives per-participant marginals and a mixed posterior predictive check (one replicate choice sequence per joint (B, O, tpar) sample, conditional on the inferred latent trajectories). Requires a fit pickle produced by the current `01_fit.py`. Writes:
  - `per_participant_settle.pkl` — first-passage time of each sampled O trajectory to the participant's true order (`settle` (S, P), T if never; `reached`), all the survival scripts need from O (replaces the 130 MB dump of the O samples themselves).
  - `hierarchical_fit.sig.json` — signature of the fit (seeds, config, hash of the tpar samples). Every pickle below embeds it, and the consumer scripts load them via `io.load_derived`, which refuses a pickle derived from a different fit than the current `hierarchical_fit.pkl` (i.e. the fit was re-run but this script was not).
  - `per_participant_marginals.pkl` — averaged belief_marg + order_marg + posterior-mean natural-scale params `(λ, ε_o, κ, κ_s, γ_o)` per ppt.
  - `per_participant_ppc.pkl` — `pred_p_wrong`, per-ppt Bayesian p-values (`p_total`, `p_swap`), error-count replicates, observed counts, swap mask; per joint sample also the replicate choice sequence (`rep_choice_packed`, 2 bits per trial) and the emission log-likelihood of the observed and of the replicated choices (`ll_obs`, `ll_rep`), from which `05_ppc.py` computes posterior predictive p-values for arbitrary test statistics.
- **`03_condition_effects.py`** — the main learnability result. Posterior of the condition means `mu[w, k]` on the natural scale for each parameter, the between-condition sd `tau_k`, ratios vs SVO with `P(ratio > 1)`, typological-group contrasts (S-/V-/O-initial), and the implied median trials-to-commit per condition (from γ_o). Figure `condition_effects.png`.
- **`04_convergence.py`** — `convergence()`: R-hat across chains on per-participant logit(λ), on the condition means and on the population means; MH acceptance per chain; per-chain modal final committed order for the TRAJ_SUBSET participants; population-λ trace (`hierarchical_fit_summary.png`). `hyperprior_predictive()`: new-participant prior vs posterior predictive per parameter on the natural scale (conditions pooled), integrating over the full hierarchy on both sides, with practical interpretations annotated (`hyperprior_predictive.png`).
- **`05_ppc.py`** — `diagnostics()`: 4 PPC figures: per-trial residual, λ decomposition, swap-trial residual by condition, late-session residual by commitment status. `pvalues()`: posterior predictive p-values `p = P(T(y_rep, θ) ≥ T(y_obs, θ))` over the joint posterior draws θ = (B, O, tpar), for: the emission log-likelihood (per participant, per condition, global), errors per 25-trial block per condition (learning-curve shape), error clustering (lapse independence), the share of swap-trial errors that land on the role-swapped scene (are errors structural?), and error rates by the number of candidates the sampled state leaves tied. It is a *mixed* PPC (replicates conditional on the inferred latent trajectories), hence conservative: it tests the emission model and the error structure, not the transition prior; per-participant p-values cluster around 0.5 under a correct model and a per-participant p near 0/1 is a strong signal. A fully marginal PPC (fresh latents from the transition prior) is deliberately not done: the HMM is a measurement model whose prior does not pull beliefs toward the true lexicon, so its marginal prediction for accuracy is chance regardless of fit quality. Figures `ppc_loglik.png`, `ppc_blocks.png`, `ppc_sequential.png`, `ppc_pvalue_calibration.png`.
- **`06_per_participant.py`** — multi-page PDF, one page per participant: order posterior trace, per-word belief lineplots, swap-trial dots, posterior-predictive P(wrong) with observed errors, correctness strip, PPC p-values.
- **`07_survival.py`** — Kaplan–Meier / log-rank and per-posterior-sample Weibull fits on the latent first-passage time of O to the true order. Kept as a model-external cross-check of `03_condition_effects.py`. Caveats: the first-passage time is only data-identified once enough of the lexicon is known (before that the transition prior fills it in), the log-rank test uses per-participant posterior medians, and Weibull medians above 200 are extrapolations.
- **`08_param_recovery.py`** — `simulate` / `fit` / `analyze` for synthetic datasets that keep the real stimuli, participants and conditions and replace the choices (`--mode prior`: truth = posterior means of the real fit's hyperparameters, per-participant tpar and latent trajectories drawn from the model, a weak test because a random-walk lexicon contradicts all four candidates on most trials, so ~77% of the simulated trials have a uniform emission vs ~19% in `posterior` mode; `--mode null`: the same with all six condition means equal, a false-positive check for the condition contrasts; `--mode posterior`: one joint posterior draw of the real fit is the truth and only the choices are regenerated, i.e. learning-like trajectories, the regime of the real data). `fit` runs `01_fit.py` in a subprocess per dataset (`EA_SIM_DATA`, `EA_RESULTS_DIR=results/param_recovery/<mode>/rep<k>/`; recovery defaults `N_OUTER=1200 BURN=400 THIN=8 N_CHAINS=3 RESUME=1`, overridable through the same env vars). `analyze` reports 95% coverage, bias, RMSE and correlation of the condition means and per-participant tpar, coverage of the hyper sds, the typological p_commit contrasts (true vs recovered, and the false-positive count in `null` mode) and latent recovery (posterior probability of the true order / belief per trial, commit-time error), and writes `figures/param_recovery_<mode>.png`. `../../slurm/slurm_param_recovery.sh` fans the fits out as a SLURM array job.

## Tests (`tests/`)

- **`test_ffbs.py`** — correctness checks for the per-word belief FFBS: (1) the analytic per-word transition table equals `_transition_lp` on all 64 candidate pairs, entry for entry, on sampled and randomly perturbed beliefs; (2) brute-force exactness: on 7-trial windows of real data all 8^6 trajectories of one word are enumerated and scored with the generic model functions, and the exact conditional is compared with the empirical distribution of `ffbs_word` draws (total variation at the Monte Carlo floor, zero mass on infeasible states). Run after any change to the belief kernel. ~2–3 min on GPU.
- **`test_simulate.py`** — checks `simulate.py` against `model.py`: simulated transitions are feasible and their frequencies match the transition probabilities; `choice_probs` equals `emit_p_correct` / `exp(emission_lp)`; the pooled profile log-likelihood of simulated participants peaks at the true parameters. ~1 min on GPU.

Both run as plain scripts (`python tests/test_ffbs.py`, prints PASS / FAIL) or under `pytest`.

## Output

- `figures/condition_effects.png` — condition-level posterior per parameter (main result)
- `figures/hierarchical_fit_summary.png` — multi-chain diagnostic
- `figures/hyperprior_predictive.png` — prior vs posterior per param
- `figures/per_participant.pdf` — per-ppt posteriors (one page per ppt)
- `figures/ppc_residual_by_trial.png`, `ppc_lambda_decomp.png`, `ppc_swap_by_order.png`, `ppc_late_by_commitment.png` — PPC calibration
- `figures/ppc_loglik.png`, `ppc_blocks.png`, `ppc_sequential.png`, `ppc_pvalue_calibration.png` — posterior predictive p-values (log-likelihood, block error counts, error clustering, swap-error structure)
- `figures/param_recovery_<mode>.png` — parameter recovery (`prior`, `null`, `posterior`)
- `figures/logrank.png`, `figures/weibull.png` — secondary survival cross-checks

## Dependencies

- `numpy`, `scipy`, `matplotlib`, `pandas` — standard
- `jax` (with CUDA support for GPU use) — pure-functional model + Gibbs inner loop
- `lifelines` — survival cross-checks
- `arviz` (+ `xarray`) — R-hat diagnostics

The conda env `pymc523` has all of these.

## To reproduce from scratch

From the repository root, once: `pip install -e .` (makes `jointlearn` importable). Then, in this folder:

```bash
python 01_fit.py                  # ~110 min on a laptop GPU at the defaults (= reported fit, 14 chains); N_CHAINS=7 halves it
python 02_postprocess.py          # minutes (post-process, no MCMC)
python 03_condition_effects.py    # main result
python 04_convergence.py
python 05_ppc.py
python 06_per_participant.py
python 07_survival.py             # secondary, ~30 min (Weibull fits over all 4200 draws)
```

Parameter recovery (needs the real fit for the ground truth; ~5 min per dataset on a laptop GPU at the recovery defaults):

```bash
python 08_param_recovery.py simulate --mode posterior --n_rep 8   # also: prior, null
python 08_param_recovery.py fit --mode posterior                  # or one dataset: --rep 3
python 08_param_recovery.py analyze --mode posterior
```

Fit pickles from before the condition-level hyperprior (single pooled
hyperprior, 5 fitted params incl. κ_s, or the older `traj_store` format) are
**not** compatible with these scripts (`io.load_fit` refuses them).

Quick end-to-end smoke test on CPU without touching `results/` or `figures/`:

```bash
export EA_RESULTS_DIR=/tmp/ea_smoke/results EA_FIGURES_DIR=/tmp/ea_smoke/figures
JAX_PLATFORMS=cpu N_OUTER=8 BURN=2 THIN=2 N_CHAINS=2 python 01_fit.py
python 02_postprocess.py && python 03_condition_effects.py   # etc.
```

For Snellius (larger config, GPU partition), see `../../slurm/README_snellius.md`.
