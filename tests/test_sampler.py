"""Checks for the HMC-within-Gibbs sampler (jointlearn/hmm/sampler.py).

  1. COUNT LIKELIHOOD — sampler.loglik_hist on the counts of sampler.hist_stats
     must equal model.loglik on the per-trial statistics, per participant, on
     realistic latents (a few sweeps from the all-UNK start) and at random
     parameters. The only difference allowed is the dropped swap term,
     n2 exp(-kappa_s) with kappa_s = 20, below float32 resolution.
  2. TARGET — the NumPyro model's log density equals the hand-written joint
     log density (priors of the non-centred parametrisation,
     plus the sum of model.loglik) at the chains' starting point.

Run:  python tests/test_sampler.py      (~1 min on GPU; prints PASS / FAIL)
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.stats import norm
from numpyro.infer.util import log_density
import numpyro.distributions as dist

from jointlearn.hmm import model as NB
from jointlearn.hmm import sampler as S


def latents(n_sweeps=5, seed=0):
    key = random.PRNGKey(seed)
    B = jnp.full((S.P_all, S.T, 7), -1, jnp.int8)
    O = jnp.full((S.P_all, S.T), S.UNK_O, jnp.int8)
    tpar = jnp.asarray(np.tile(S.HYPER_INIT[:, 0], (S.P_all, 1)), jnp.float32)
    for _ in range(n_sweeps):
        key, B, O = S.sweep_all(key, B, O, tpar)
    return B, O


def test_sampler():
    B, O = latents()
    st = S.stats_all(B, O)
    hs = vmap(S.hist_stats)(*st)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(5):
        tpar = jnp.asarray(S.HYPER_INIT[:, 0] + rng.normal(0, 1, (S.P_all, S.K)), jnp.float32)
        nat = NB.to_nat(tpar)
        a = vmap(NB.loglik)(nat, *st)
        b = vmap(S.loglik_hist)(nat, *hs)
        worst = max(worst, float(jnp.max(jnp.abs(a - b) / jnp.abs(a))))
    ok1 = worst < 1e-5
    print(f'1. count likelihood vs model.loglik: {"PASS" if ok1 else "FAIL"} '
          f'(max relative difference {worst:.1e} over {S.P_all} participants x 5 draws)')

    ini = S.start_values(S.SEED0)
    v = {k: ini[k] for k in ('m', 'mu_raw', 'z')}
    v['tau2'], v['sigma2'] = jnp.exp(ini['tau2']), jnp.exp(ini['sigma2'])
    v.update(S.with_stats(B, O))
    lp_model = float(log_density(S.model, (), {}, v)[0])
    mu = v['m'] + jnp.sqrt(v['tau2']) * v['mu_raw']
    tpar = mu[S.COND_J] + jnp.sqrt(v['sigma2']) * v['z']
    lp_hand = float(norm.logpdf(v['m'], S.MU0, S.SD0).sum()
                    + dist.InverseGamma(S.A_TAU, S.B_TAU).log_prob(v['tau2']).sum()
                    + dist.InverseGamma(S.A0, S.B0).log_prob(v['sigma2']).sum()
                    + norm.logpdf(v['mu_raw']).sum() + norm.logpdf(v['z']).sum()
                    + vmap(NB.loglik)(NB.to_nat(tpar), *st).sum())
    rel = abs(lp_model - lp_hand) / abs(lp_hand)
    ok2 = rel < 1e-5
    print(f'2. NumPyro target vs hand-written joint density: {"PASS" if ok2 else "FAIL"} '
          f'({lp_model:.2f} vs {lp_hand:.2f})')
    assert ok1 and ok2
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    test_sampler()
