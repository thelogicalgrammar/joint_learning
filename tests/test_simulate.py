"""Correctness checks for simulate.py against model.py.

  1. Every simulated belief transition is feasible under model._transition_lp
     (finite log-prob) and the simulated move frequencies match the model's
     transition probabilities: the per-trial probability of a single move is
     n_single e^-kappa / Z, so the total number of single moves is compared
     with its expectation +- 3 sd; the order chain's commit / switch counts
     are compared with p_commit_o / eps_o the same way.
  2. simulate.choice_probs agrees with model.emit_p_correct (candidate 0) and
     with exp(model.emission_lp) for every candidate, bit for bit, on
     simulated and on posterior-like trajectories.
  3. The data log-likelihood of a simulated participant is maximised near the
     true parameters: over 60 simulated participants, the pooled profile
     log-likelihood in each transformed coordinate (others at truth) peaks
     within one grid step of the truth.

Run:  python tests/test_simulate.py    (~1 min on GPU; prints PASS / FAIL)
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from jointlearn.hmm import model as NB
from jointlearn.hmm import simulate as SIM
from jointlearn.hmm import sampler as FH
from jointlearn.hmm import config as CFG


def test_simulate():
    U, C, WO = np.asarray(FH.U), np.asarray(FH.C), np.asarray(FH.WO)
    P, T = U.shape[:2]
    rng = np.random.default_rng(0)
    tpar_true = np.array([np.log(0.08 / 0.92), np.log(3.0), np.log(0.02 / 0.98),
                          np.log(0.05 / 0.95)], np.float32)
    lam, kappa, eps_o, p_commit = NB.tpar_to_nat_np(tpar_true)
    NP = 60
    idx = rng.choice(P, NP, replace=False)

    B = np.zeros((NP, T, NB.N_W), np.int8)
    O = np.zeros((NP, T), np.int8)
    for i, p in enumerate(idx):
        B[i], O[i] = SIM.simulate_latents(rng, tpar_true, U[p])
    CH = SIM.simulate_choices(rng, B, O, np.full(NP, lam, np.float32), U[idx], C[idx], WO)

    # ---- 1. transitions feasible + move frequencies ----
    stats = jax.jit(vmap(NB.stats_arrays, in_axes=(0, 0, 0, 0, 0, None)))(
        jnp.asarray(B), jnp.asarray(O), jnp.asarray(U[idx]), jnp.asarray(C[idx]),
        jnp.asarray(CH, jnp.int32), jnp.asarray(WO))
    nS, surv, mt, nsa, nwa, n_unk_stay, n_commit, n_o_stay, n_o_switch = map(np.asarray, stats)
    tr_lp = np.asarray(jax.jit(vmap(vmap(NB._transition_lp, in_axes=(0, 0, 0, None, None)),
                                    in_axes=(0, 0, 0, None, None)))(
        jnp.asarray(B[:, :-1]), jnp.asarray(B[:, 1:]), jnp.asarray(U[idx, 1:]),
        jnp.float32(kappa), jnp.float32(NB.KAPPA_S_FIXED)))
    ok1 = np.isfinite(tr_lp).all() and (mt >= 0).all()
    Z = 1.0 + nsa * np.exp(-kappa) + nwa * np.exp(-NB.KAPPA_S_FIXED)
    p_single = (nsa * np.exp(-kappa) / Z)[:, 1:]
    exp_single, sd_single = p_single.sum(), np.sqrt((p_single * (1 - p_single)).sum())
    n_single = (mt == 1).sum()
    z_single = (n_single - exp_single) / sd_single
    n_unk = n_unk_stay.sum() + n_commit.sum()
    z_commit = (n_commit.sum() - n_unk * p_commit) / np.sqrt(n_unk * p_commit * (1 - p_commit))
    n_com = n_o_stay.sum() + n_o_switch.sum()
    z_switch = (n_o_switch.sum() - n_com * eps_o) / np.sqrt(n_com * eps_o * (1 - eps_o))
    ok1 &= max(abs(z_single), abs(z_commit), abs(z_switch)) < 3
    print(f'1. transitions feasible: {bool(np.isfinite(tr_lp).all())}; single moves '
          f'{n_single} vs E {exp_single:.0f} (z={z_single:+.2f}); commits {n_commit.sum()} '
          f'(z={z_commit:+.2f}); switches {n_o_switch.sum()} (z={z_switch:+.2f}): '
          f'{"PASS" if ok1 else "FAIL"}')

    # ---- 2. choice_probs == emission functions ----
    def check_probs(B_, O_, lam_, U_, C_):
        pr = np.asarray(SIM.choice_probs(jnp.asarray(B_), jnp.asarray(O_),
                                         jnp.asarray(lam_, jnp.float32), jnp.asarray(U_),
                                         jnp.asarray(C_), jnp.asarray(WO)))
        pc = np.asarray(jax.jit(vmap(lambda b, o, u, c, l: vmap(
            NB.emit_p_correct, in_axes=(0, 0, 0, 0, None, None))(o, b, u, c, l, jnp.asarray(WO))))(
            jnp.asarray(B_), jnp.asarray(O_), jnp.asarray(U_), jnp.asarray(C_),
            jnp.asarray(lam_, jnp.float32)))
        e_all = np.asarray(jax.jit(vmap(lambda b, o, u, c, l: vmap(
            lambda o_t, b_t, u_t, c_t: vmap(NB.emission_lp, in_axes=(None, None, None, None, 0, None, None))(
                o_t, b_t, u_t, c_t, jnp.arange(4), l, jnp.asarray(WO)))(o, b, u, c)))(
            jnp.asarray(B_), jnp.asarray(O_), jnp.asarray(U_), jnp.asarray(C_),
            jnp.asarray(lam_, jnp.float32)))
        return (np.array_equal(pr[..., 0], pc), np.abs(np.exp(e_all) - pr).max(),
                np.abs(pr.sum(-1) - 1).max())
    eq0, d_all, d_sum = check_probs(B, O, np.full(NP, lam, np.float32), U[idx], C[idx])
    # posterior-like trajectories: partially known lexicon converging to the truth
    B2 = np.full((NP, T, NB.N_W), -1, np.int8)
    O2 = np.full((NP, T), NB.UNK_O, np.int8)
    from jointlearn.hmm import dataset as RP
    for i, p in enumerate(idx):
        for w in range(NB.N_W):
            B2[i, rng.integers(5, T):, w] = RP.inv_true[p, w]
        O2[i, rng.integers(5, T):] = RP.true_order[p]
    lam2 = rng.uniform(0.01, 0.3, NP).astype(np.float32)
    eq0b, d_allb, d_sumb = check_probs(B2, O2, lam2, U[idx], C[idx])
    ok2 = eq0 and eq0b and max(d_all, d_allb) < 1e-6 and max(d_sum, d_sumb) < 1e-5
    print(f'2. choice_probs[..., 0] == emit_p_correct bitwise: {eq0 and eq0b}; '
          f'max |exp(emission_lp) - choice_probs| {max(d_all, d_allb):.1e}; rows sum to 1 within '
          f'{max(d_sum, d_sumb):.1e}: {"PASS" if ok2 else "FAIL"}')

    # ---- 3. profile log-likelihood peaks at the truth ----
    ll_fn = jax.jit(vmap(lambda t, *st: NB.loglik(NB.to_nat(t), *st), in_axes=(None,) + (0,) * 9))
    grid = np.linspace(-1.0, 1.0, 21)
    ok3 = True
    for k, name in enumerate(CFG.PARAM_NAMES):
        prof = []
        for d in grid:
            t = tpar_true.copy(); t[k] += d
            prof.append(float(ll_fn(jnp.asarray(t), *stats).sum()))
        i = int(np.argmax(prof))
        best = grid[i]
        # standard error from the profile curvature at the peak (finite second
        # difference); the peak must sit within 2 se + one grid step of the truth
        i = min(max(i, 1), len(grid) - 2)
        curv = (prof[i + 1] - 2 * prof[i] + prof[i - 1]) / (grid[1] - grid[0]) ** 2
        se = 1.0 / np.sqrt(max(-curv, 1e-9))
        ok3 &= abs(best) <= 2 * se + (grid[1] - grid[0]) + 1e-9
        print(f'   {name:>17}: pooled profile log-lik peaks at truth {best:+.1f} '
              f'(grid step 0.1, se {se:.2f})')
    print(f'3. profile log-likelihood peaks within 2 se + one grid step of the truth '
          f'for all {len(CFG.PARAM_NAMES)} coordinates: {"PASS" if ok3 else "FAIL"}')
    assert ok1 and ok2 and ok3
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    test_simulate()
