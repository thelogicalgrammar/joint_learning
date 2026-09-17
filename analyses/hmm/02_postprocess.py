"""
Post-process the hierarchical fit into per-participant marginals, joint
O samples for the survival cross-checks, and posterior predictive replicates.

Reads:  results/hierarchical_fit.pkl
        (contains, per chain: hyper, tpar, B, O — full joint posterior samples)

Writes: results/per_participant_settle.pkl     (first-passage times of O to the
                                                 true order, per joint sample)
        results/per_participant_marginals.pkl   (averaged belief_marg + order_marg)
        results/per_participant_ppc.pkl         (pred_p_wrong, p-values, replicate
                                                 choices, emission log-liks)
        results/hierarchical_fit.sig.json       (signature of the fit these came from)

Every pickle carries the fit's signature under io.SIG_KEY; the consumer
scripts load them through io.load_derived, which refuses a pickle
derived from a different fit than the current hierarchical_fit.pkl.

The PPC integrates over the joint posterior: each retained (B, O, tpar) is
one joint draw from p(B, O, tpar | data) and one replicate choice sequence
is drawn per sample. Note it is a *mixed* posterior predictive check —
replicates are conditional on the latent trajectories inferred from the
observed choices — which is conservative (the latents have already
absorbed some misfit). Per sample, the replicate choice sequence itself
(2-bit packed) and the emission log-likelihood of the observed and of the
replicated choices are stored, so that 05_ppc.py can compute
posterior predictive p-values for any test statistic — including the
likelihood itself — without touching the fit again.

This script runs no MCMC — it just summarises the fit pickle (seconds).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from jointlearn.hmm import model as NB
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm import simulate as SIM


# ---- load the hierarchical fit ----
fit = IO.load_fit()
P_all, T = RP.P, RP.T
print(f'Loaded {len(fit["chains"])} chains', flush=True)
FIT_SIG = IO.write_fit_signature(fit)

# Concatenate samples across chains: (S_total, P, T, 7), etc.
B_all = IO.pool_chains(fit, 'B')            # (S_total, P, T, 7) int8
O_all = IO.pool_chains(fit, 'O')            # (S_total, P, T) int8
tpar_all = IO.pool_chains(fit, 'tpar')      # (S_total, P, N_TPAR) float32
S_total = B_all.shape[0]
print(f'Joint posterior: S_total = {S_total} samples per participant', flush=True)


# ---- batched data (constant across samples) ----
U = jnp.asarray(np.ascontiguousarray(RP.signals.transpose(1, 0, 2)), dtype=jnp.int32)
C = jnp.asarray(np.ascontiguousarray(RP.cand_mean.transpose(1, 0, 2, 3)), dtype=jnp.int32)
WO = jnp.asarray(RP.word_orders, dtype=jnp.int32)
CH = np.asarray(RP.choices.T)                                          # (P, T)

# ---- swap mask (P, T) bool — role-swap (word-order-discriminating) trials ----
SWAP_MASK_PT = AN.compute_swap_mask().T                               # (P_all, T)
ERR_OBS = (CH != 0).sum(axis=1)                                        # (P,)
ERR_OBS_SWAP = ((CH != 0) & SWAP_MASK_PT).sum(axis=1)                  # (P,)


# ---- per-trial choice distribution over all participants ----
@jax.jit
def choice_probs_all(B, O, params):
    """Per (ppt, trial) probability of each of the 4 candidates under the
    emission model (simulate.choice_probs; column 0 = Pr(correct), identical
    to model.emit_p_correct). B (P, T, 7), O (P, T), params (P, 5) natural.
    Returns (P, T, 4)."""
    return SIM.choice_probs(B, O, params[:, 0], U, C, WO)


# ---- accumulate marginals + PPC over the joint posterior ----
print('Accumulating per-ppt marginals + PPC from joint samples...', flush=True)
t0 = time.time()
bc_total = np.zeros((P_all, T, 7, 8), dtype=np.float32)       # belief count
oc_total = np.zeros((P_all, T, 7), dtype=np.float32)          # order count
pc_total = np.zeros((P_all, T), dtype=np.float32)             # Pr(correct) mean
n_err_rep = np.zeros((S_total, P_all), dtype=np.int16)
n_err_rep_swap = np.zeros((S_total, P_all), dtype=np.int16)
ll_obs = np.zeros((S_total, P_all), dtype=np.float32)   # emission log-lik of the data
ll_rep = np.zeros((S_total, P_all), dtype=np.float32)   # ... of the replicate
rep_choice = np.zeros((S_total, P_all, (T + 3) // 4), dtype=np.uint8)  # 2-bit packed
# per (sample, ppt, trial): number of candidates the sampled state leaves
# tied at the top (1..4; 4 = uniform emission: no committed heard word, no
# surviving candidate, or all four survive), stored 2-bit packed as nS - 1.
n_surv = np.zeros((S_total, P_all, (T + 3) // 4), dtype=np.uint8)
rng = np.random.default_rng(0)

p_ax = np.arange(P_all)[:, None, None]
t_ax = np.arange(T)[None, :, None]
s_ax = np.arange(7)[None, None, :]
p_ax_2d = np.arange(P_all)[:, None]
t_ax_2d = np.arange(T)[None, :]
for s in range(S_total):
    B_s, O_s = B_all[s], O_all[s]                                      # (P, T, 7), (P, T)
    params_s = np.asarray(NB.to_nat(tpar_all[s]))                      # (P, 5) natural

    # belief / order marginals
    v = np.where(B_s == -1, 7, B_s)                                    # (P, T, 7) 0..7
    np.add.at(bc_total, (p_ax, t_ax, s_ax, v), 1.0)
    np.add.at(oc_total, (p_ax_2d, t_ax_2d, O_s), 1.0)

    # PPC: per-trial choice distribution given the sampled latents. One
    # uniform per trial picks the replicate choice by inverse CDF; the
    # replicate is wrong iff u > Pr(correct), so the error counts are the
    # same as a Bernoulli draw with this uniform.
    pr = np.asarray(choice_probs_all(jnp.asarray(B_s), jnp.asarray(O_s),
                                     jnp.asarray(params_s, dtype=jnp.float32)))  # (P, T, 4)
    pc = pr[..., 0]
    pc_total += pc
    u = rng.random(pc.shape)
    err_rep = u > pc
    ch_rep = np.minimum((u[..., None] > np.cumsum(pr, axis=-1)).sum(-1), 3)
    n_err_rep[s] = err_rep.sum(axis=1)
    n_err_rep_swap[s] = (err_rep & SWAP_MASK_PT).sum(axis=1)
    lp = np.log(pr)
    ll_obs[s] = np.take_along_axis(lp, CH[..., None], -1)[..., 0].sum(axis=1)
    ll_rep[s] = np.take_along_axis(lp, ch_rep[..., None], -1)[..., 0].sum(axis=1)
    rep_choice[s] = AN.pack_choices(ch_rep)
    n_surv[s] = AN.pack_choices((pr >= pr.max(-1, keepdims=True) - 1e-6).sum(-1) - 1)

    if (s + 1) % 50 == 0 or s == S_total - 1:
        print(f'  {s+1}/{S_total}  ({time.time()-t0:.0f}s)', flush=True)

bc_total /= S_total
oc_total /= S_total
pc_total /= S_total
print(f'Joint-posterior summarization done in {(time.time() - t0):.1f}s.\n')


# ---- write the three downstream pickles ----
# First-passage times of each sampled O trajectory to the participant's true
# order (T if never): all the survival cross-checks need from O.
settle, reached = AN.settle_matrix(O_all, RP.true_order)              # (S_total, P)
IO.save_pickle({'settle': settle, 'reached': reached, 'T': T,
                 IO.SIG_KEY: FIT_SIG},
                'per_participant_settle.pkl')

# per-ppt natural-scale params (lam, eps_o, kappa, kappa_s, p_commit_o) at
# the posterior-mean transformed values (≈ natural-scale posterior median;
# kappa_s is the fixed constant)
params_mean = np.asarray(NB.to_nat(tpar_all.mean(axis=0)))             # (P, 5)
marginals = {p: {'belief_marg': bc_total[p],
                 'order_marg': oc_total[p],
                 'params': params_mean[p]}
             for p in range(P_all)}
marginals[IO.SIG_KEY] = FIT_SIG
IO.save_pickle(marginals, 'per_participant_marginals.pkl')

p_total = (n_err_rep >= ERR_OBS[None, :]).mean(axis=0)
p_swap = (n_err_rep_swap >= ERR_OBS_SWAP[None, :]).mean(axis=0)
IO.save_pickle({
    'pred_p_wrong': 1.0 - pc_total,                                    # (P, T)
    'err_obs': ERR_OBS,
    'err_obs_swap': ERR_OBS_SWAP,
    'n_err_rep': n_err_rep,
    'n_err_rep_swap': n_err_rep_swap,
    'p_total': p_total,
    'p_swap': p_swap,
    'swap_mask': SWAP_MASK_PT,
    'll_obs': ll_obs,                                                   # (S, P)
    'll_rep': ll_rep,                                                   # (S, P)
    'rep_choice_packed': rep_choice,                                    # (S, P, T/4) 2-bit
    'n_surv_packed': n_surv,                                            # (S, P, T/4) 2-bit, nS - 1
    'T': T,
    IO.SIG_KEY: FIT_SIG,
}, 'per_participant_ppc.pkl')
p_ll = (ll_rep >= ll_obs).mean(axis=0)
print(f'  median p_total = {np.median(p_total):.3f}, '
      f'median p_swap = {np.median(p_swap):.3f}, median p_loglik = {np.median(p_ll):.3f}')
print(f'  global log-likelihood p-value P(ll_rep >= ll_obs) = '
      f'{(ll_rep.sum(1) >= ll_obs.sum(1)).mean():.3f}  (see 05_ppc.py)')
print(f'  participants with extreme p_total (<0.025 or >0.975): '
      f'{((p_total < 0.025) | (p_total > 0.975)).sum()}/{P_all}')
print(f'  participants with extreme p_swap  (<0.025 or >0.975): '
      f'{((p_swap < 0.025) | (p_swap > 0.975)).sum()}/{P_all}')
