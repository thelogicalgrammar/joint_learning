"""
HMC-within-Gibbs sampler for the hierarchical partial-lexicon HMM (library
part of the fit; the entry point is analyses/hmm/01_fit.py).

Target: all participants jointly, with a CONDITION-LEVEL hyperprior
(condition = the participant's word order, 6 levels; see config.py):

  tpar[p, k] ~ N(mu[w(p), k], sigma_k^2),  mu[w, k] ~ N(m_k, tau_k^2),
  m_k ~ N(mu0_k, sd0_k^2),  sigma_k^2 ~ IG(A0, B0),  tau_k^2 ~ IG(A_TAU, B_TAU)

with 4 fitted transformed parameters per participant
  t[0] = logit(lam), t[1] = log(kappa), t[2] = logit(eps_o), t[3] = logit(p_commit_o)
(kappa_s is fixed at model.KAPPA_S_FIXED; it is unidentified, see README).

One iteration (NumPyro HMCGibbs):
  1. Gibbs: every participant's latent belief and order trajectories (B, O)
     are redrawn given its parameters by model.traj_sweep (exact FFBS),
     vmapped over participants, and summarised into sufficient statistics.
  2. HMC: all continuous parameters (m, tau^2, sigma^2, condition means and
     participant parameters, non-centred: mu = m + tau * mu_raw,
     tpar = mu[w] + sigma * z) jointly, given those statistics. Fixed-length
     HMC with the trajectory length redrawn every iteration uniformly in
     [0.5, 1.5] x TRAJ (JitterHMC); step size and diagonal mass matrix are
     adapted during warm-up. NUTS gives the same posterior but its tree
     building costs ~0.4 ms per leapfrog step here, ~8x the gradient.

Sufficient statistics: the likelihood (model.loglik) depends on the per-trial
statistics of model.stats_arrays only through how often each value occurs,
so they are collapsed into 40 counts per participant (hist_stats) and the
HMC target uses loglik_hist. The swap term n2 exp(-kappa_s) (kappa_s = 20)
is below float32 resolution next to 1 + n1 exp(-kappa) and is dropped.

Chains differ by seed and by a dispersed start (start_values); the latent
trajectories always start all-UNK. Note: chains cannot be initialised at
specific orders, since traj_sweep resamples O exactly at every sweep.
Several chains in one process exhaust a 4 GB GPU, so 01_fit.py runs each
chain in its own process (run_chain) and assembles them (save_fit).
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import xlogy
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, HMC, HMCGibbs

from jointlearn.hmm import model as NB
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO
from jointlearn.hmm import config as CFG
from jointlearn.hmm.config import (HYPER_INIT, TAU_INIT, PRIOR_MU0, PRIOR_SD0,
                                   A0, B0, A_TAU, B_TAU)

# -------- run config (env vars override; defaults = the reported fit of
# 2026-11-10: 8 chains x (1000 warm-up + 6000 iterations, thin 20) = 300
# retained draws per chain, 2400 pooled; ~15 min per chain on a laptop
# RTX 3050 Ti) --------
N_WARMUP = int(os.environ.get('N_WARMUP', 1000))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 6000))    # post-warm-up iterations
THIN = int(os.environ.get('THIN', 20))
N_CHAINS = int(os.environ.get('N_CHAINS', 8))
TRAJ = float(os.environ.get('TRAJ', 3.3))             # base HMC trajectory length
SEED0 = int(os.environ.get('SEED0', 20000))           # chain k has seed SEED0 + k
# RESUME=1: 01_fit.py reuses finished chain files (results/hier_chain{k}.pkl)
# whose config matches; unfinished chains are rerun from the start.
RESUME = os.environ.get('RESUME', '0') == '1'
assert N_WARMUP >= 0 and N_SAMPLES >= 1 and THIN >= 1, \
    f'need N_WARMUP >= 0, N_SAMPLES >= 1, THIN >= 1 (got {N_WARMUP}, {N_SAMPLES}, {THIN})'

N_TPAR = K = CFG.N_TPAR                          # = 4 fitted params
assert N_TPAR == NB.N_TPAR == HYPER_INIT.shape[0]
UNK_O = NB.UNK_O
SAMPLER = 'HMC-within-Gibbs (NumPyro HMCGibbs, jittered fixed-length HMC)'

# -------- batched data --------
P_all, T = RP.signals.shape[1], RP.signals.shape[0]
PARTS = np.arange(P_all)
COND = np.asarray(RP.true_order, dtype=np.int64)     # (P,) condition index 0..5
N_COND = W = len(RP.order_labels)
U = jnp.asarray(np.ascontiguousarray(RP.signals.transpose(1, 0, 2)), dtype=jnp.int32)
C = jnp.asarray(np.ascontiguousarray(RP.cand_mean.transpose(1, 0, 2, 3)), dtype=jnp.int32)
CH = jnp.asarray(np.ascontiguousarray(RP.choices.T), dtype=jnp.int32)
WO = jnp.asarray(RP.word_orders, dtype=jnp.int32)
COND_J = jnp.asarray(COND)
MU0 = jnp.asarray(PRIOR_MU0, jnp.float32)
SD0 = jnp.asarray(PRIOR_SD0, jnp.float32)


# -------- latent trajectories --------

@jax.jit
def sweep_all(key, B, O, tpar):
    """One Gibbs sweep of (B, O) for all participants given tpar (P, K).
    B (P, T, 7) int8, O (P, T) int8. Returns (key, B_new, O_new)."""
    keys = random.split(key, P_all + 1)
    _, B_new, O_new = vmap(NB.traj_sweep, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))(
        keys[1:], B, O, NB.to_nat(tpar), U, C, CH, WO, 1)
    return keys[0], B_new, O_new


@jax.jit
def stats_all(B, O):
    """model.stats_arrays for all participants (6-tuple, leading P axis)."""
    return vmap(NB.stats_arrays, in_axes=(0, 0, 0, 0, 0, None))(B, O, U, C, CH, WO)


# -------- likelihood from counts --------

N1 = 25          # n1 (number of available single moves) takes values 0..24


def hist_stats(nS, surv, mt, n1, n2, nO):
    """One participant's stats_arrays collapsed into counts:
    em (4, 2)  trials by number of surviving candidates (1..4) x chosen survives,
    mtc (3,)   number of single moves, swaps, impossible transitions,
    n1h (N1,)  trials by number of available single moves,
    oc (4,)    order transitions UNK->UNK, UNK->order, order stays, order switches."""
    em = jnp.zeros((4, 2)).at[nS.astype(jnp.int32) - 1, surv.astype(jnp.int32)].add(1.0)
    mtc = jnp.stack([(mt == 1).sum(), (mt == 2).sum(), (mt == -1).sum()]).astype(jnp.float32)
    n1h = jnp.zeros(N1).at[n1.astype(jnp.int32)].add(1.0)
    comm = nO[:NB.N_O, :NB.N_O]
    oc = jnp.stack([nO[UNK_O, UNK_O], nO[UNK_O, :NB.N_O].sum(),
                    jnp.trace(comm), comm.sum() - jnp.trace(comm)])
    return em, mtc, n1h, oc


def loglik_hist(params, em, mtc, n1h, oc):
    """model.loglik from the counts of hist_stats (natural-scale params)."""
    lam, eps_o, kappa, kappa_s, p_commit_o = params
    nS = jnp.arange(1, 5, dtype=jnp.float32)[:, None]
    sv = jnp.arange(2, dtype=jnp.float32)[None, :]
    lp_em = xlogy(em, NB.emit_p(nS, sv, lam)).sum()
    lp_mv = (-mtc[0] * kappa - mtc[1] * kappa_s + jnp.where(mtc[2] > 0, -jnp.inf, 0.0)
             - (n1h * jnp.log1p(jnp.arange(N1) * jnp.exp(-kappa))).sum())
    lp_o = (xlogy(oc[0], 1 - p_commit_o) + xlogy(oc[1], p_commit_o / NB.N_O)
            + xlogy(oc[2], 1 - eps_o) + xlogy(oc[3], eps_o / (NB.N_O - 1)))
    return lp_em + lp_mv + lp_o


STATS = ('em', 'mtc', 'n1h', 'oc')
# Gibbs sites: latents (stored as float, B shifted by +1) and their counts
GIBBS_SHAPES = dict(B=(P_all, T, 7), O=(P_all, T), em=(P_all, 4, 2), mtc=(P_all, 3),
                    n1h=(P_all, N1), oc=(P_all, 4))


def with_stats(B, O):
    """Gibbs-site dict for latents (B, O)."""
    st = vmap(hist_stats)(*stats_all(B, O))
    d = {'B': B.astype(jnp.float32) + 1, 'O': O.astype(jnp.float32)}
    d.update({n: x.astype(jnp.float32) for n, x in zip(STATS, st)})
    return d


# -------- NumPyro model --------

def tpar_from(h):
    """Participant parameters (P, K) from the non-centred HMC sites."""
    mu = h['m'] + jnp.sqrt(h['tau2']) * h['mu_raw']
    return mu[COND_J] + jnp.sqrt(h['sigma2']) * h['z']


def model():
    m = numpyro.sample('m', dist.Normal(MU0, SD0).to_event(1))
    tau2 = numpyro.sample('tau2', dist.InverseGamma(A_TAU, B_TAU).expand([K]).to_event(1))
    sigma2 = numpyro.sample('sigma2', dist.InverseGamma(A0, B0).expand([K]).to_event(1))
    mu_raw = numpyro.sample('mu_raw', dist.Normal(0., 1.).expand([W, K]).to_event(2))
    z = numpyro.sample('z', dist.Normal(0., 1.).expand([P_all, K]).to_event(2))
    mu = m + jnp.sqrt(tau2) * mu_raw
    tpar = mu[COND_J] + jnp.sqrt(sigma2) * z
    numpyro.deterministic('mu', mu)
    numpyro.deterministic('tpar', tpar)
    site = {n: numpyro.sample(n, dist.ImproperUniform(constraints.real, (), s))
            for n, s in GIBBS_SHAPES.items()}
    numpyro.factor('loglik', vmap(loglik_hist)(NB.to_nat(tpar), *[site[n] for n in STATS]).sum())


def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
    """Redraw (B, O) for all participants given the HMC sites."""
    keys = random.split(rng_key, P_all)
    _, B, O = vmap(NB.traj_sweep, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))(
        keys, jnp.round(gibbs_sites['B'] - 1).astype(jnp.int8),
        jnp.round(gibbs_sites['O']).astype(jnp.int8), NB.to_nat(tpar_from(hmc_sites)),
        U, C, CH, WO, 1)
    return with_stats(B, O)


class JitterHMC(HMC):
    """HMC whose trajectory length is redrawn every iteration, independently of
    the state (so the kernel stays reversible): uniform on [0.5, 1.5] x base."""
    def __init__(self, model, base, **kw):
        super().__init__(model, trajectory_length=base, **kw)
        self._base = base

    def sample(self, state, model_args, model_kwargs):
        u = random.uniform(random.fold_in(state.rng_key, 7), minval=0.5, maxval=1.5)
        return super().sample(state._replace(trajectory_length=self._base * u),
                              model_args, model_kwargs)


def start_values(seed):
    """Dispersed starting point (unconstrained scale for tau2, sigma2), latents all-UNK."""
    r = np.random.default_rng(seed)
    B0_ = jnp.full((P_all, T, 7), -1, jnp.int8)
    O0_ = jnp.full((P_all, T), UNK_O, jnp.int8)
    tau2 = jnp.log(jnp.full(K, TAU_INIT ** 2))
    sigma2 = jnp.log(jnp.asarray(HYPER_INIT[:, 1] ** 2, jnp.float32))
    ini = with_stats(B0_, O0_)
    ini['m'] = MU0 + jnp.asarray(r.normal(0, 0.5, (K,)), jnp.float32)
    ini['mu_raw'] = jnp.asarray(r.normal(0, 1, (W, K)), jnp.float32)
    ini['z'] = jnp.asarray(r.normal(0, 1, (P_all, K)), jnp.float32)
    ini['tau2'] = tau2 + jnp.asarray(r.normal(0, 0.5, (K,)), jnp.float32)
    ini['sigma2'] = sigma2 + jnp.asarray(r.normal(0, 0.5, (K,)), jnp.float32)
    return ini


# -------- one chain --------

def config_tuple():
    """Everything that defines the target or the kernel; a chain file with a
    different config is not reused by RESUME."""
    def _t(a):
        return tuple(float(x) for x in np.ravel(a))
    return (SAMPLER, N_WARMUP, N_SAMPLES, THIN, TRAJ, P_all, T, N_TPAR,
            tuple(int(c) for c in COND), _t(HYPER_INIT), float(TAU_INIT),
            _t(PRIOR_MU0), _t(PRIOR_SD0), float(A0), float(B0), float(A_TAU),
            float(B_TAU), float(NB.KAPPA_S_FIXED))


def chain_path(out_dir, k):
    return Path(out_dir) / f'hier_chain{k}.pkl'


def load_chain(out_dir, k):
    """Finished chain k from out_dir if it exists and matches the current config, else None."""
    p = chain_path(out_dir, k)
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        ch = pickle.load(f)
    if ch.get('config') != config_tuple() or ch.get('seed') != SEED0 + k:
        print(f'  {p.name}: other config or seed, not reused', flush=True)
        return None
    return ch


FIELDS = ('hmc_state.num_steps', 'hmc_state.diverging', 'hmc_state.adapt_state.step_size')


def run_chain(k):
    """Run chain k (seed SEED0 + k) and return it as a dict in the fit format."""
    seed = SEED0 + k
    ini = start_values(seed)
    kernel = HMCGibbs(JitterHMC(model, TRAJ), gibbs_fn=gibbs_fn, gibbs_sites=list(GIBBS_SHAPES))
    # progress_bar=True segfaults with this JAX/NumPyro combination
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES, thinning=THIN,
                progress_bar=False)
    t0 = time.time()
    mcmc.warmup(random.PRNGKey(seed), init_params=ini, extra_fields=FIELDS,
                collect_warmup=False)
    jax.block_until_ready(mcmc.post_warmup_state)
    t1 = time.time()
    # init_params again: without it run() fails with "invalid log_factor"
    mcmc.run(mcmc.post_warmup_state.rng_key, init_params=ini, extra_fields=FIELDS)
    smp = mcmc.get_samples()
    jax.block_until_ready(smp['mu'])
    t2 = time.time()
    ex = mcmc.get_extra_fields()
    hmc = dict(steps=np.asarray(ex['hmc_state.num_steps']),
               div=np.asarray(ex['hmc_state.diverging']),
               step_size=np.asarray(ex['hmc_state.adapt_state.step_size']),
               warm_secs=t1 - t0, samp_secs=t2 - t1)
    ch = dict(chain_id=k, seed=seed, it=N_SAMPLES, config=config_tuple(),
              hyper=dict(mu=np.asarray(smp['mu']), sigma=np.sqrt(np.asarray(smp['sigma2'])),
                         m=np.asarray(smp['m']), tau=np.sqrt(np.asarray(smp['tau2']))),
              tpar=np.asarray(smp['tpar']),
              B=(np.rint(np.asarray(smp['B'])) - 1).astype(np.int8),
              O=np.rint(np.asarray(smp['O'])).astype(np.int8),
              hmc=hmc, parts=PARTS, acc=RP.accuracy[PARTS])
    print(f'[chain {k}] seed {seed}: warm-up {t1 - t0:.0f}s, sampling {t2 - t1:.0f}s '
          f'(both incl. compile), {ch["tpar"].shape[0]} draws, mean leapfrog steps '
          f'{hmc["steps"].mean():.0f}, step size {hmc["step_size"][-1]:.4f}, '
          f'divergences {int(hmc["div"].sum())}', flush=True)
    return ch


def save_chain(ch, out_dir):
    p = chain_path(out_dir, ch['chain_id'])
    with open(p, 'wb') as f:
        pickle.dump(ch, f)
    return p


def save_fit(chains, out_path):
    """Write the fit pickle and its signature sidecar (io.write_fit_signature),
    against which the pickles written by 02_postprocess.py are checked for
    staleness. n_outer / burn count iterations including warm-up."""
    fit = dict(chains=chains, n_outer=N_WARMUP + N_SAMPLES, burn=N_WARMUP, thin=THIN,
               n_warmup=N_WARMUP, n_samples=N_SAMPLES, traj=TRAJ, sampler=SAMPLER,
               n_tpar=N_TPAR, param_names=CFG.PARAM_NAMES,
               parts=PARTS, acc=RP.accuracy[PARTS],
               cond=COND, cond_labels=list(RP.order_labels),
               # prior config, saved so downstream scripts don't duplicate it
               hyper_init=HYPER_INIT, tau_init=TAU_INIT,
               prior_mu0=PRIOR_MU0, prior_sd0=PRIOR_SD0,
               a0=A0, b0=B0, a_tau=A_TAU, b_tau=B_TAU,
               kappa_s_fixed=NB.KAPPA_S_FIXED)
    with open(str(out_path), 'wb') as f:
        pickle.dump(fit, f)
    IO.write_fit_signature(fit, Path(out_path).parent)
