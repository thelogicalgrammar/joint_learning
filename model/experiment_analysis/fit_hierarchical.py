"""
Hierarchical fit using the partial-lexicon HMM with UNK-order extension,
with a CONDITION-LEVEL hyperprior (condition = the participant's word
order, 6 levels). JAX vmap over all participants on GPU; Python loop over
N_CHAINS independent chains that differ only by PRNG seed. Note: chains
CANNOT be initialised at specific orders — traj_sweep resamples O exactly
(FFBS given B) at the start of every sweep, so any O init is immediately
overwritten. Convergence diagnostics therefore rest on multi-seed
agreement.

Per-participant model: 4 fitted transformed parameters
  t[0] = logit(lam), t[1] = log(kappa), t[2] = logit(eps_o),
  t[3] = logit(p_commit_o)
(kappa_s is fixed at model.KAPPA_S_FIXED; it is unidentified — see README).

Hierarchy (see config.py):
  tpar[p, k] ~ N(mu[w(p), k], sigma_k^2),  mu[w, k] ~ N(m_k, tau_k^2),
  m_k ~ N(mu0_k, sd0_k^2),  sigma_k^2 ~ IG(A0, B0),  tau_k^2 ~ IG(A_TAU, B_TAU)

Condition effects on learnability are read from the posterior of mu[:, k]
(analyze_condition_effects.py), replacing the earlier post-hoc survival
analysis on latent first-passage times.

Writes a per-chain checkpoint every CKPT_EVERY iters so partial results can
be analysed without waiting for the full run.
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random, vmap

import model as NB
import data as RP
import helpers as HLP
import config as CFG
from config import (HYPER_INIT, TAU_INIT, PRIOR_MU0, PRIOR_SD0,
                    A0, B0, A_TAU, B_TAU, PROP_SD, SHIFT_SD, SHIFT_SD_POP)

# -------- config (env vars override; defaults = local-run values) --------
N_OUTER = int(os.environ.get('N_OUTER', 600))
BURN = int(os.environ.get('BURN', 200))
THIN = int(os.environ.get('THIN', 4))
TRAJ_SUBSET = [42, 102, 247, 63, 213, 250, 114]      # ppts shown in the summary figure
N_CHAINS = int(os.environ.get('N_CHAINS', 7))         # independent seeds
N_FFBS = int(os.environ.get('N_FFBS', 1))           # per-word belief FFBS passes / sweep
N_BLOCK = int(os.environ.get('N_BLOCK', 0))         # legacy block-move MH (0 = off)
N_SWAP = int(os.environ.get('N_SWAP', 0))          # swap MH (0 = off; see traj_sweep)
CKPT_EVERY = int(os.environ.get('CKPT_EVERY', 100))
# RESUME=1: reuse finished chains and continue interrupted ones from their
# last checkpoint (results/hier_chain{k}_ckpt.pkl). Checkpoints carry the
# full sampler state, so a resumed chain is identical to an uninterrupted one.
RESUME = os.environ.get('RESUME', '0') == '1'
assert 0 <= BURN < N_OUTER and THIN >= 1, \
    f'need 0 <= BURN < N_OUTER and THIN >= 1 (got N_OUTER={N_OUTER}, BURN={BURN}, THIN={THIN})'

N_TPAR = CFG.N_TPAR                              # = 4 fitted params
assert N_TPAR == NB.N_TPAR == HYPER_INIT.shape[0]
UNK_O = NB.UNK_O

# -------- batched data --------
P_all, T = RP.signals.shape[1], RP.signals.shape[0]
PARTS = np.arange(P_all)
COND = np.asarray(RP.true_order, dtype=np.int64)     # (P,) condition index 0..5
N_COND = len(RP.order_labels)
U = jnp.asarray(np.ascontiguousarray(RP.signals.transpose(1, 0, 2)), dtype=jnp.int32)
C = jnp.asarray(np.ascontiguousarray(RP.cand_mean.transpose(1, 0, 2, 3)), dtype=jnp.int32)
CH = jnp.asarray(np.ascontiguousarray(RP.choices.T), dtype=jnp.int32)
HT_np = np.full((P_all, 7, T), -1, dtype=np.int32)
HC_np = np.zeros((P_all, 7), dtype=np.int32)
for p in range(P_all):
    Up = np.asarray(U[p])
    for s in range(7):
        idx = np.where(np.any(Up == s, axis=1))[0]
        HT_np[p, s, :len(idx)] = idx
        HC_np[p, s] = len(idx)
HT = jnp.asarray(HT_np)
HC = jnp.asarray(HC_np)
WO = jnp.asarray(RP.word_orders, dtype=jnp.int32)


# -------- per-ppt traj sweep (vmap'd) --------

@jax.jit
def sweep_all(key, B, O, tpar):
    """One vmap'd Gibbs sweep across all P_all participants.
    Inputs:  B (P, T, 7) int8, O (P, T) int8, tpar (P, N_TPAR) float32
    Returns: (key, B_new, O_new)
    """
    keys = random.split(key, P_all + 1)
    nat = NB.to_nat(tpar)                                        # (P, 5)
    _, B_new, O_new = vmap(NB.traj_sweep,
                           in_axes=(0, 0, 0, 0, 0, 0, 0, None,
                                    0, 0, None, None, None))(
        keys[1:], B, O, nat, U, C, CH, WO, HT, HC, N_BLOCK, N_SWAP, N_FFBS)
    return keys[0], B_new, O_new


@jax.jit
def stats_all(B, O):
    """vmap'd stats_arrays across P_all participants. Returns a 9-tuple of
    arrays each with a leading P axis."""
    return vmap(NB.stats_arrays, in_axes=(0, 0, 0, 0, 0, None))(
        B, O, U, C, CH, WO)


# -------- per-ppt MH on transformed params --------

def _lp_one(tpar_i, stats_i, mu_i, sigma):
    """log p(data | tpar_i) + log N(tpar_i | mu_i, sigma^2), per participant.
    tpar_i, mu_i: (N_TPAR,) — mu_i is the participant's CONDITION mean;
    sigma: (N_TPAR,) within-condition sd (shared across conditions)."""
    ll = NB.loglik(NB.to_nat(tpar_i), *stats_i)
    diffs = (tpar_i - mu_i) / sigma
    pr = (-0.5 * diffs ** 2 - jnp.log(sigma)).sum()
    return ll + pr


PROP_SD_J = jnp.asarray(PROP_SD, dtype=jnp.float32)


def _mh4_one(keys4, tpar_i, stats_i, mu_i, sigma):
    """Coordinate-wise MH on tpar_i: one random-walk proposal per transformed
    coordinate k = 0..N_TPAR-1 in turn, with proposal sd PROP_SD[k]. Returns
    the updated tpar_i and a (N_TPAR,) vector of accept indicators.

    Why coordinate-wise: the four coordinates have posterior widths that
    differ by more than an order of magnitude (log_kappa ~0.08 vs
    logit_p_commit ~1.2). A joint isotropic proposal (the previous kernel,
    sd 0.25 on all four) was vetoed by the tight coordinate and crawled on
    the wide one (lag-4 autocorrelation 0.90 on logit_p_commit). Updating
    one coordinate at a time with its own scale removes both problems.

    Proposal symmetry: each step is a Gaussian random walk on one coordinate,
    q(t' | t) = q(t | t'), so the accept ratio reduces to
    log_u < new_lp - old_lp with no Hastings correction.
    """
    cur = _lp_one(tpar_i, stats_i, mu_i, sigma)

    def step(carry, inp):
        k, key = inp
        tpar, cur = carry
        k_prop, k_acc = random.split(key)
        prop = tpar.at[k].add(PROP_SD_J[k] * random.normal(k_prop))
        new = _lp_one(prop, stats_i, mu_i, sigma)
        log_u = jnp.log(random.uniform(k_acc))
        accept = log_u < (new - cur)
        return (jnp.where(accept, prop, tpar),
                jnp.where(accept, new, cur)), accept.astype(jnp.int32)
    (tpar_new, _), acc = lax.scan(step, (tpar_i, cur),
                                  (jnp.arange(N_TPAR), keys4))
    return tpar_new, acc


def _ll_one(tpar_i, stats_i):
    """Data log-likelihood only (no hierarchy term), per participant."""
    return NB.loglik(NB.to_nat(tpar_i), *stats_i)


COND_J = jnp.asarray(COND)
SHIFT_SD_J = jnp.asarray(SHIFT_SD, dtype=jnp.float32)
SHIFT_SD_POP_J = jnp.asarray(SHIFT_SD_POP, dtype=jnp.float32)
PRIOR_MU0_J = jnp.asarray(PRIOR_MU0, dtype=jnp.float32)
PRIOR_SD0_J = jnp.asarray(PRIOR_SD0, dtype=jnp.float32)


@jax.jit
def shift_all(key, tpar, stats, mu, m, tau, sd=SHIFT_SD_J, sd_pop=SHIFT_SD_POP_J):
    """Group shift move: for each coordinate k in turn and each condition w
    (in parallel; conditions touch disjoint participants), propose adding
    one delta ~ N(0, sd[k]^2) to mu[w, k] AND to tpar[p, k] for every
    participant p in condition w. Accept/reject per (w, k).

    Why: eps_o (and to a lesser degree p_commit) participants are
    data-starved, so tpar_p | mu_w sits at mu_w + noise and mu_w | tpar sits
    at mean(tpar) +- sigma/sqrt(n): the alternating conjugate scheme moves a
    condition mean by ~sigma/sqrt(n) per iteration while its marginal is
    several times wider (lag-4 acf ~0.9 on the eps_o condition means, ESS
    ~3 per 600 draws). A common shift relocates the whole group at once.

    Accept ratio: the within-condition Gaussian terms are invariant under a
    common shift (tpar_p - mu_w unchanged), so only the participants'
    data log-likelihood changes and the prior N(mu_w | m, tau) enter.
    The proposal is symmetric (delta vs -delta), no Hastings correction.

    Second stage, population level: the same coupling exists one level up
    (m | mu has sd tau/sqrt(W) while its marginal is far wider for the
    data-starved coordinates), so for each k also propose one delta ~
    N(0, sd_pop[k]^2) added to m[k], every mu[w, k] and every tpar[p, k].
    Now the mu | m terms cancel as well; only the total likelihood change
    and the prior N(m | mu0, sd0) enter.

    tpar (P, K), stats leaf-wise (P, ...), mu (W, K), m (K,), tau (K,),
    sd, sd_pop (K,). Returns (key, tpar_new, mu_new, m_new, acc (W, K),
    acc_pop (K,)) with 0/1 accept indicators.
    """
    ll_cur = vmap(_ll_one)(tpar, stats)                                # (P,)

    def step(carry, inp):
        k, key = inp
        tpar, mu, ll_cur = carry
        k_prop, k_acc = random.split(key)
        delta = sd[k] * random.normal(k_prop, (N_COND,))              # (W,)
        tpar_prop = tpar.at[:, k].add(delta[COND_J])
        ll_prop = vmap(_ll_one)(tpar_prop, stats)                      # (P,)
        dll = jax.ops.segment_sum(ll_prop - ll_cur, COND_J,
                                  num_segments=N_COND)                 # (W,)
        mu_k, mu_prop_k = mu[:, k], mu[:, k] + delta
        dprior = (-0.5 * ((mu_prop_k - m[k]) / tau[k]) ** 2
                  + 0.5 * ((mu_k - m[k]) / tau[k]) ** 2)
        accept = jnp.log(random.uniform(k_acc, (N_COND,))) < dll + dprior
        acc_p = accept[COND_J]
        tpar = tpar.at[:, k].set(jnp.where(acc_p, tpar_prop[:, k], tpar[:, k]))
        mu = mu.at[:, k].set(jnp.where(accept, mu_prop_k, mu_k))
        ll_cur = jnp.where(acc_p, ll_prop, ll_cur)
        return (tpar, mu, ll_cur), accept.astype(jnp.int32)
    keys = random.split(key, 2 * N_TPAR + 1)
    (tpar, mu, ll_cur), acc = lax.scan(step, (tpar, mu, ll_cur),
                                       (jnp.arange(N_TPAR), keys[1:N_TPAR + 1]))

    def step_pop(carry, inp):
        k, key = inp
        tpar, mu, m, ll_cur = carry
        k_prop, k_acc = random.split(key)
        delta = sd_pop[k] * random.normal(k_prop)
        tpar_prop = tpar.at[:, k].add(delta)
        ll_prop = vmap(_ll_one)(tpar_prop, stats)
        m_prop = m[k] + delta
        dprior = (-0.5 * ((m_prop - PRIOR_MU0_J[k]) / PRIOR_SD0_J[k]) ** 2
                  + 0.5 * ((m[k] - PRIOR_MU0_J[k]) / PRIOR_SD0_J[k]) ** 2)
        accept = jnp.log(random.uniform(k_acc)) < (ll_prop - ll_cur).sum() + dprior
        tpar = jnp.where(accept, tpar_prop, tpar)
        mu = mu.at[:, k].set(jnp.where(accept, mu[:, k] + delta, mu[:, k]))
        m = m.at[k].set(jnp.where(accept, m_prop, m[k]))
        ll_cur = jnp.where(accept, ll_prop, ll_cur)
        return (tpar, mu, m, ll_cur), accept.astype(jnp.int32)
    (tpar, mu, m, _), acc_pop = lax.scan(step_pop, (tpar, mu, m, ll_cur),
                                         (jnp.arange(N_TPAR), keys[N_TPAR + 1:]))
    return keys[0], tpar, mu, m, acc.T, acc_pop                       # (W,K), (K,)


@jax.jit
def mh_all(key, tpar, stats, mu_p, sigma):
    """vmap _mh4_one across participants.
    tpar (P, N_TPAR), stats leaf-wise (P, ...), mu_p (P, N_TPAR) = condition
    mean per participant, sigma (N_TPAR,). Returns (key, tpar_new, n_acc)
    with n_acc (P, N_TPAR) accept indicators."""
    keys = random.split(key, P_all * N_TPAR + 1).reshape(-1, 2)
    mh_keys = keys[1:].reshape(P_all, N_TPAR, 2)
    tpar_new, n_acc = vmap(_mh4_one, in_axes=(0, 0, 0, 0, None))(
        mh_keys, tpar, stats, mu_p, sigma)
    return keys[0], tpar_new, n_acc


# -------- conjugate hyper update (numpy) --------

def init_hyper():
    """Initial hyper state: every condition mean at HYPER_INIT mu."""
    return dict(mu=np.tile(HYPER_INIT[:, 0], (N_COND, 1)),   # (W, K)
                sigma=HYPER_INIT[:, 1].copy(),                # (K,)
                m=HYPER_INIT[:, 0].copy(),                    # (K,)
                tau=np.full(N_TPAR, TAU_INIT))                # (K,)


def update_hyper(x, cond, hyp, rng, mu0=PRIOR_MU0, sd0=PRIOR_SD0,
                 a0=A0, b0=B0, a_tau=A_TAU, b_tau=B_TAU):
    """One Gibbs pass over the hyperparameters given x = tpar (P, K) and
    condition index cond (P,). Every block is conjugate:

      mu[w]   | rest ~ N( (sum_{p in w} x_p / sigma^2 + m / tau^2) / prec,
                          1 / prec ),   prec = n_w / sigma^2 + 1 / tau^2
      sigma^2 | rest ~ IG( a0 + P/2,  b0 + 0.5 * sum_p (x_p - mu[w(p)])^2 )
      m       | rest ~ N( (sum_w mu[w] / tau^2 + mu0 / sd0^2) / prec_m,
                          1 / prec_m ), prec_m = W / tau^2 + 1 / sd0^2
      tau^2   | rest ~ IG( a_tau + W/2,  b_tau + 0.5 * sum_w (mu[w] - m)^2 )

    All operations are vectorised over the K parameter dimensions.
    """
    mu, sigma, m, tau = hyp['mu'].copy(), hyp['sigma'], hyp['m'], hyp['tau']
    P, K = x.shape
    W = mu.shape[0]
    # condition means
    for w in range(W):
        xw = x[cond == w]
        n = len(xw)
        prec = n / sigma ** 2 + 1.0 / tau ** 2
        mean = (xw.sum(axis=0) / sigma ** 2 + m / tau ** 2) / prec
        mu[w] = rng.normal(mean, 1.0 / np.sqrt(prec))
    # within-condition sd (shared across conditions)
    resid = x - mu[cond]
    a_n = a0 + P / 2.0
    b_n = b0 + 0.5 * (resid ** 2).sum(axis=0)
    sigma = np.sqrt(1.0 / rng.gamma(a_n, 1.0 / b_n))
    # population mean
    prec_m = W / tau ** 2 + 1.0 / sd0 ** 2
    mean_m = (mu.sum(axis=0) / tau ** 2 + mu0 / sd0 ** 2) / prec_m
    m = rng.normal(mean_m, 1.0 / np.sqrt(prec_m))
    # between-condition sd
    a_t = a_tau + W / 2.0
    b_t = b_tau + 0.5 * ((mu - m[None, :]) ** 2).sum(axis=0)
    tau = np.sqrt(1.0 / rng.gamma(a_t, 1.0 / b_t))
    return dict(mu=mu, sigma=sigma, m=m, tau=tau)


# -------- one chain --------

def _load_ckpt(ckpt_path, seed):
    """Return a usable checkpoint dict for this chain, or None. A checkpoint
    is usable if RESUME is set, it exists, was written by this code version
    (carries `state`), has the same seed and config (kernel settings, priors,
    fixed constants, data), and is not further along than N_OUTER."""
    if not RESUME or not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    cfg = ck.get('config')
    if 'state' not in ck or ck.get('seed') != seed or cfg != _config_tuple():
        print(f'  checkpoint {os.path.basename(ckpt_path)} not resumable '
              f'(old format, other seed, or other config/prior) — restarting chain',
              flush=True)
        return None
    if ck['it'] > N_OUTER:
        # The retained-sample buffers are sized from N_OUTER and the sampler
        # state is at outer iteration ck['it']; a longer chain cannot be
        # truncated back, so refuse rather than crash or mis-normalise.
        raise SystemExit(
            f'checkpoint {os.path.basename(ckpt_path)} is at outer iteration '
            f'{ck["it"]} > N_OUTER={N_OUTER}. Re-run with N_OUTER >= {ck["it"]} '
            f'to reuse it, or with RESUME=0 to restart the chain.')
    return ck


def fmt_acc(rate):
    """Per-coordinate MH acceptance rates as 'a/b/c/d'."""
    return '/'.join(f'{r:.2f}' for r in np.atleast_1d(rate))


def _config_tuple():
    # Everything that defines the posterior target or the kernel, so that a
    # checkpoint sampled under a different prior / model / data is refused.
    # N_OUTER is deliberately not included: a finished chain can be extended
    # by re-running with RESUME=1 and a larger N_OUTER (see _load_ckpt).
    def _t(a):
        return tuple(float(x) for x in np.ravel(a))
    return (BURN, THIN, N_BLOCK, N_SWAP, N_FFBS, P_all, T, N_TPAR,
            tuple(int(c) for c in COND),
            _t(PROP_SD), _t(SHIFT_SD), _t(SHIFT_SD_POP),
            _t(HYPER_INIT), float(TAU_INIT), _t(PRIOR_MU0), _t(PRIOR_SD0),
            float(A0), float(B0), float(A_TAU), float(B_TAU),
            float(NB.KAPPA_S_FIXED))


def run_chain(chain_id, seed, ckpt_path):
    rng = np.random.default_rng(seed)
    key = random.PRNGKey(seed)

    # state (the O init is irrelevant — the first sweep's FFBS replaces it)
    B = jnp.full((P_all, T, 7), -1, dtype=jnp.int8)
    O = jnp.full((P_all, T), UNK_O, dtype=jnp.int8)
    tpar = jnp.asarray(np.tile(HYPER_INIT[:, 0], (P_all, 1)),
                       dtype=jnp.float32)                    # (P, N_TPAR)
    hyp = init_hyper()

    # The record condition below fires ceil((N_OUTER - BURN) / THIN) times.
    n_samp = -(-(N_OUTER - BURN) // THIN)
    hyper_samp = dict(mu=np.zeros((n_samp, N_COND, N_TPAR), dtype=np.float32),
                      sigma=np.zeros((n_samp, N_TPAR), dtype=np.float32),
                      m=np.zeros((n_samp, N_TPAR), dtype=np.float32),
                      tau=np.zeros((n_samp, N_TPAR), dtype=np.float32))
    tpar_samp = np.zeros((n_samp, P_all, N_TPAR), dtype=np.float32)
    # Full per-ppt (B, O) joint samples.
    # Storage: ~46 MB (B) + ~6.5 MB (O) per chain at default config.
    B_samp = np.zeros((n_samp, P_all, T, 7), dtype=np.int8)
    O_samp = np.zeros((n_samp, P_all, T), dtype=np.int8)
    acc_count = np.zeros(N_TPAR, dtype=np.int64)    # MH acceptances per coord
    shift_count = np.zeros(N_TPAR, dtype=np.int64)  # group-shift acceptances per coord
    pshift_count = np.zeros(N_TPAR, dtype=np.int64) # population-shift acceptances per coord
    si = 0
    it0 = 0

    # ---- resume from a checkpoint written by this code version ----
    ck = _load_ckpt(ckpt_path, seed)
    if ck is not None:
        st = ck['state']
        B, O = jnp.asarray(st['B']), jnp.asarray(st['O'])
        tpar = jnp.asarray(st['tpar'], dtype=jnp.float32)
        hyp = st['hyp']
        rng.bit_generator.state = st['rng_state']
        key = jnp.asarray(st['key'], dtype=jnp.uint32)
        acc_count = st['acc_count']
        shift_count = st['shift_count']
        pshift_count = st['pshift_count']
        si = ck['tpar'].shape[0]
        for k in hyper_samp:
            hyper_samp[k][:si] = ck['hyper'][k]
        tpar_samp[:si], B_samp[:si], O_samp[:si] = ck['tpar'], ck['B'], ck['O']
        it0 = ck['it']
        print(f'[chain {chain_id}] resuming from checkpoint at outer {it0}/{N_OUTER} '
              f'({si} samples)', flush=True)

    t0 = time.time()

    def _chain_dict(it, with_state=False):
        d = dict(chain_id=chain_id, seed=seed, it=it, config=_config_tuple(),
                 hyper={k: v[:si] for k, v in hyper_samp.items()},
                 tpar=tpar_samp[:si], B=B_samp[:si], O=O_samp[:si],
                 mh_accept_rate=acc_count / max(it * P_all, 1),   # (N_TPAR,)
                 shift_accept_rate=shift_count / max(it * N_COND, 1),  # (N_TPAR,)
                 pop_shift_accept_rate=pshift_count / max(it, 1),        # (N_TPAR,)
                 parts=PARTS, acc=RP.accuracy[PARTS])
        if with_state:      # everything needed to continue the chain exactly
            d['state'] = dict(B=np.asarray(B), O=np.asarray(O),
                              tpar=np.asarray(tpar), hyp=hyp,
                              rng_state=rng.bit_generator.state,
                              key=np.asarray(key), acc_count=acc_count,
                              shift_count=shift_count, pshift_count=pshift_count)
        return d

    if it0 >= N_OUTER:
        print(f'[chain {chain_id}] already complete — reusing checkpoint', flush=True)
        return _chain_dict(N_OUTER)

    for it in range(it0, N_OUTER):
        # 1) Gibbs sweep B, O (vmap'd)
        key, B, O = sweep_all(key, B, O, tpar)
        # 2) stats (vmap'd)
        stats = stats_all(B, O)
        # 3) MH on tpar (vmap'd) given each ppt's condition mean
        mu_p = jnp.asarray(hyp['mu'][COND], dtype=jnp.float32)
        key, tpar, n_acc = mh_all(key, tpar, stats, mu_p,
                                  jnp.asarray(hyp['sigma'], dtype=jnp.float32))
        acc_count += np.asarray(n_acc).sum(axis=0)
        # 3b) group shift move on (mu[w, k], tpar[p in w, k])
        key, tpar, mu_new, m_new, s_acc, p_acc = shift_all(
            key, tpar, stats, jnp.asarray(hyp['mu'], dtype=jnp.float32),
            jnp.asarray(hyp['m'], dtype=jnp.float32),
            jnp.asarray(hyp['tau'], dtype=jnp.float32))
        hyp['mu'] = np.asarray(mu_new, dtype=np.float64)
        hyp['m'] = np.asarray(m_new, dtype=np.float64)
        shift_count += np.asarray(s_acc).sum(axis=0)
        pshift_count += np.asarray(p_acc)
        # 4) conjugate hyper update (numpy)
        tpar_np = np.asarray(tpar)
        hyp = update_hyper(tpar_np, COND, hyp, rng)
        # 5) record
        if it >= BURN and (it - BURN) % THIN == 0:
            for k in hyper_samp:
                hyper_samp[k][si] = hyp[k]
            tpar_samp[si] = tpar_np
            B_samp[si] = np.asarray(B)
            O_samp[si] = np.asarray(O)
            si += 1
        # 6) progress + checkpoint
        if (it + 1) % 25 == 0:
            B.block_until_ready()
            el = time.time() - t0
            eta = el / (it + 1 - it0) * (N_OUTER - it - 1) / 60
            print(f'[chain {chain_id}] outer {it+1}/{N_OUTER}  '
                  f'({el:.0f}s, eta {eta:.1f}m, MH acc '
                  f'{fmt_acc(acc_count / ((it + 1) * P_all))}, shift acc '
                  f'{fmt_acc(shift_count / ((it + 1) * N_COND))}, pop '
                  f'{fmt_acc(pshift_count / (it + 1))})', flush=True)
        if (it + 1) % CKPT_EVERY == 0 or it + 1 == N_OUTER:
            with open(ckpt_path, 'wb') as f:
                pickle.dump(_chain_dict(it + 1, with_state=True), f)

    return _chain_dict(N_OUTER)


def save_fit(chains, out_path):
    """Write the fit pickle (also used by the smoke test) and its signature
    sidecar (helpers.write_fit_signature), against which the pickles written
    by infer_per_participant.py are checked for staleness."""
    fit = dict(chains=chains, n_outer=N_OUTER, burn=BURN, thin=THIN,
               n_tpar=N_TPAR, param_names=CFG.PARAM_NAMES,
               parts=PARTS, acc=RP.accuracy[PARTS],
               cond=COND, cond_labels=list(RP.order_labels),
               traj_subset=TRAJ_SUBSET,
               # prior config — saved so downstream scripts don't
               # have to duplicate it.
               hyper_init=HYPER_INIT, tau_init=TAU_INIT,
               prior_mu0=PRIOR_MU0, prior_sd0=PRIOR_SD0,
               a0=A0, b0=B0, a_tau=A_TAU, b_tau=B_TAU,
               kappa_s_fixed=NB.KAPPA_S_FIXED)
    with open(str(out_path), 'wb') as f:
        pickle.dump(fit, f)
    HLP.write_fit_signature(fit, Path(out_path).parent)


def print_condition_means(chains):
    """Posterior-median condition means on the natural scale, per chain."""
    print('\n=== condition means (posterior median over samples) per chain ===')
    names = ['lambda', 'kappa', 'eps_o', 'p_commit']
    for ch in chains:
        mu = ch['hyper']['mu']                                 # (S, W, K)
        print(f'chain {ch["chain_id"]} (seed {ch["seed"]}, MH acc '
              f'{fmt_acc(ch["mh_accept_rate"])}, shift acc '
              f'{fmt_acc(ch.get("shift_accept_rate", np.nan))}, pop '
              f'{fmt_acc(ch.get("pop_shift_accept_rate", np.nan))})')
        print(f'  {"cond":>5}  ' + '  '.join(f'{n:>8}' for n in names))
        for w, lbl in enumerate(RP.order_labels):
            nat = np.asarray(NB.to_nat(jnp.asarray(np.median(mu[:, w], axis=0))))
            lam, eps, kap, _, pc = nat
            print(f'  {lbl:>5}  {lam:>8.3f}  {kap:>8.2f}  {eps:>8.4f}  {pc:>8.4f}')


if __name__ == '__main__':
    out_dir = HLP.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Hierarchical fit (condition-level hyperprior), JAX vmap on '
          f'{jax.devices()[0]}.')
    print(f'  {N_CHAINS} chains x {N_OUTER} outer iters x {P_all} participants '
          f'x {N_COND} conditions (N_TPAR={N_TPAR}, kappa_s fixed at '
          f'{NB.KAPPA_S_FIXED})', flush=True)

    print('  compiling sweep_all + stats_all + mh_all (warm-up) ...',
          flush=True)
    t_c = time.time()
    # (named *_warm, not B0/O0: `B0` is the InvGamma scale imported from
    # config and used as a default argument of update_hyper.)
    key_warm = random.PRNGKey(0)
    B_warm = jnp.full((P_all, T, 7), -1, dtype=jnp.int8)
    O_warm = jnp.full((P_all, T), UNK_O, dtype=jnp.int8)
    tpar_warm = jnp.asarray(np.tile(HYPER_INIT[:, 0], (P_all, 1)),
                            dtype=jnp.float32)
    hyp_warm = init_hyper()
    k1, B1, O1 = sweep_all(key_warm, B_warm, O_warm, tpar_warm)
    s1 = stats_all(B1, O1)
    _ = mh_all(k1, tpar_warm, s1,
               jnp.asarray(hyp_warm['mu'][COND], dtype=jnp.float32),
               jnp.asarray(hyp_warm['sigma'], dtype=jnp.float32))
    print(f'    compiled in {time.time() - t_c:.1f}s', flush=True)

    t0 = time.time()
    chains = []
    for k in range(N_CHAINS):
        ck_path = str(out_dir / f'hier_chain{k}_ckpt.pkl')
        ch = run_chain(chain_id=k, seed=10000 + 100 * k, ckpt_path=ck_path)
        chains.append(ch)
    print(f'\nAll chains done in {(time.time() - t0) / 60:.1f} min', flush=True)

    out = out_dir / HLP.FIT_NAME
    save_fit(chains, out)
    print(f'Saved: {out}')
    print_condition_means(chains)
    print('done')
