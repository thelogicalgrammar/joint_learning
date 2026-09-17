"""Correctness checks for the per-word belief FFBS (model.ffbs_word).

Self-contained: builds a realistic sampler state from the real data (a few
sweeps from the all-UNK init at the prior-mean parameters), then

  1. TABLE EQUALITY — model._word_transition_table (the analytic per-word
     transition table) must equal _transition_lp evaluated on all 64
     candidate pairs, entry for entry, for every word, on the sampled
     beliefs and on randomly perturbed beliefs (which contain change
     patterns real chains never produce: 3+ simultaneous changes, non-swap
     double changes, ...), at random kappa / kappa_s.
  2. EXACTNESS — on short windows of real trials, enumerate ALL 8^(Tw-1)
     trajectories of one word (t = 0 pinned), score them with the generic
     emission / transition functions plus the bijection mask, and compare
     that exact conditional with the empirical distribution of ffbs_word
     draws. Total variation must be at the Monte Carlo noise floor and no
     draw may land on an infeasible state.

Run:  python tests/test_ffbs.py      (~2-3 min on GPU; prints PASS / FAIL)
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap

from jointlearn.hmm import model as NB
from jointlearn.hmm import sampler as FH
from jointlearn.hmm.model import (emission_lp, _transition_lp, _word_transition_table,
                   N_W, N_M, N_BS, UNK_O)


def test_ffbs():
    P, T, WO = FH.P_all, FH.T, FH.WO
    U, C, CH = np.asarray(FH.U), np.asarray(FH.C), np.asarray(FH.CH)
    rng = np.random.default_rng(0)

    # ---- realistic state: 30 sweeps from all-UNK at the prior-mean params ----
    tpar = jnp.asarray(np.tile(FH.HYPER_INIT[:, 0], (P, 1)), dtype=jnp.float32)
    B = jnp.full((P, T, N_W), -1, jnp.int8)
    O = jnp.full((P, T), UNK_O, jnp.int8)
    key = random.PRNGKey(0)
    for _ in range(30):
        key, B, O = FH.sweep_all(key, B, O, tpar)
    nat = NB.to_nat(tpar)
    B_np, O_np = np.asarray(B), np.asarray(O)

    # ---- 0. the JAX and numpy parameter transforms agree (model.TPAR_SPEC) ----
    _t = rng.normal(0, 3, (500, NB.N_TPAR)).astype(np.float32)
    _nat_jax = np.asarray(NB.to_nat(jnp.asarray(_t)))[:, list(NB.TPAR_NAT_INDEX)]
    _nat_np = NB.tpar_to_nat_np(_t)
    _nat_np_k = np.stack([NB.tpar_to_nat_np(_t[:, k], k) for k in range(NB.N_TPAR)], -1)
    assert np.allclose(_nat_jax, _nat_np, rtol=1e-5, atol=1e-7) and np.array_equal(_nat_np, _nat_np_k)
    assert np.all(np.asarray(NB.to_nat(jnp.asarray(_t)))[:, NB.I_KAPPA_S] == NB.KAPPA_S_FIXED)
    print('0. to_nat (JAX) == tpar_to_nat_np (numpy) on 500 random tpar: PASS')
    print(f'state: {P} ppts, {(B_np != -1).mean():.2f} of belief cells committed')

    # ---- 1. table equality ----
    def generic_table(s, B1, U1, kappa, kappa_s):
        vals = jnp.arange(N_BS, dtype=B1.dtype) - 1
        Bc = vmap(lambda b_t: vmap(lambda v: b_t.at[s].set(v))(vals))(B1)
        return vmap(lambda bp_, bc_, u: vmap(lambda bp: vmap(
            lambda bc: _transition_lp(bp, bc, u, kappa, kappa_s))(bc_))(bp_))(
            Bc[:-1], Bc[1:], U1[1:])

    ok, n_entries = True, 0
    for s in range(N_W):
        f_an = jax.jit(vmap(lambda B1, U1, p: _word_transition_table(s, B1, U1, p[2], p[3])))
        f_ge = jax.jit(vmap(lambda B1, U1, p: generic_table(s, B1, U1, p[2], p[3])))
        for rep in range(3):
            k1, k2, k3 = random.split(random.PRNGKey(100 * s + rep), 3)
            Bt = B if rep == 0 else jnp.where(
                random.uniform(k1, B.shape) < 0.08,
                random.randint(k2, B.shape, -1, N_M).astype(B.dtype), B)
            p = nat.at[:, 2].set(jnp.exp(random.normal(k3, (P,)))).at[:, 3].set(2.0 + rep)
            a, g = f_an(Bt, FH.U, p), f_ge(Bt, FH.U, p)
            ok &= bool(jnp.array_equal(a, g))
            n_entries += a.size
    print(f'1. analytic table == generic table on {n_entries / 1e6:.0f}M entries: '
          f'{"PASS" if ok else "FAIL"}')
    assert ok

    # ---- 2. brute-force exactness on windows ----
    def exact_dist(s, Bw, Ow, Uw, Cw, CHw, params):
        Tw = Bw.shape[0]
        lam, _, kappa, kappa_s, _ = params
        grid = np.array(list(itertools.product(range(N_BS), repeat=Tw - 1)), np.int32)
        X = np.concatenate([np.full((len(grid), 1), int(Bw[0, s]) + 1, np.int32), grid], 1)
        vals = jnp.arange(N_BS, dtype=jnp.int8) - 1
        Bj, Oj, Uj, Cj, CHj = map(jnp.asarray, (Bw, Ow, Uw, Cw, CHw))
        others = jnp.arange(N_W) != s
        taken = ((Bj[:, :, None] == jnp.arange(N_M)[None, None, :])
                 & others[None, :, None]).any(1)                          # (Tw, N_M)

        def score(x):
            Bx = Bj.at[:, s].set(vals[x])
            em = vmap(emission_lp, in_axes=(0, 0, 0, 0, 0, None, None))(
                Oj, Bx, Uj, Cj, CHj, lam, WO)
            bad = (x > 0) & taken[jnp.arange(Tw), jnp.maximum(x - 1, 0)]
            em = jnp.where(bad, -jnp.inf, em)
            tr = vmap(lambda bp, bc, u: _transition_lp(bp, bc, u, kappa, kappa_s))(
                Bx[:-1], Bx[1:], Uj[1:])
            return em.sum() + tr.sum()
        lp = np.asarray(jax.jit(vmap(score))(jnp.asarray(X)))
        p = np.exp(lp - lp.max())
        return X, p / p.sum()

    def window_case(p_idx, s, t0, Tw, params, n_draw=200_000, seed=0):
        sl = slice(t0, t0 + Tw)
        Bw, Ow, Uw, Cw, CHw = B_np[p_idx, sl], O_np[p_idx, sl], U[p_idx, sl], C[p_idx, sl], CH[p_idx, sl]
        X, p = exact_dist(s, Bw, Ow, Uw, Cw, CHw, params)
        draw = jax.jit(vmap(lambda k: NB.ffbs_word(
            k, s, jnp.asarray(Bw), jnp.asarray(Ow), jnp.asarray(Uw), jnp.asarray(Cw),
            jnp.asarray(CHw), WO, params)[:, s]))
        xs = np.asarray(draw(random.split(random.PRNGKey(seed), n_draw))).astype(np.int32) + 1
        assert (xs[:, 0] == int(Bw[0, s]) + 1).all(), 't = 0 was resampled'
        w = (N_BS ** np.arange(Tw - 1))[::-1]
        K = N_BS ** (Tw - 1)
        emp = np.bincount((xs[:, 1:] * w).sum(1), minlength=K) / n_draw
        exact = np.zeros(K); exact[(X[:, 1:] * w).sum(1)] = p
        tv = 0.5 * np.abs(emp - exact).sum()
        h1 = np.bincount((xs[: n_draw // 2, 1:] * w).sum(1), minlength=K) / (n_draw // 2)
        h2 = np.bincount((xs[n_draw // 2:, 1:] * w).sum(1), minlength=K) / (n_draw // 2)
        tv_mc = 0.5 * np.abs(h1 - h2).sum() / np.sqrt(2)          # MC noise floor
        infeasible = emp[exact == 0].sum()
        print(f'   ppt {p_idx:3d} word {s} trials {t0}-{t0 + Tw - 1}: support {int((p > 1e-6).sum()):5d} '
              f'| TV(exact, draws) {tv:.4f} vs MC floor {tv_mc:.4f} | mass on infeasible {infeasible:.0e}')
        return tv, tv_mc, infeasible

    acc = np.asarray(FH.RP.accuracy)
    hi = np.argsort(-acc)[:60]
    cases = []
    while len(cases) < 8:
        p_idx, t0, Tw = int(rng.choice(hi)), int(rng.integers(30, T - 8)), 7
        for s in rng.permutation(N_W):
            if (U[p_idx, t0:t0 + Tw] == s).any(1).sum() >= 3 and (B_np[p_idx, t0] != -1).sum() >= 3:
                cases.append((p_idx, int(s), t0, Tw))
                break
    print('2. brute-force exactness on 7-trial windows (8^6 trajectories each):')
    res = []
    for i, (p_idx, s, t0, Tw) in enumerate(cases):
        params = jnp.array([rng.uniform(0.05, 0.3), 0.01, rng.uniform(0.2, 1.2),
                            rng.uniform(1.0, 4.0), 0.05], jnp.float32)
        res.append(window_case(p_idx, s, t0, Tw, params, seed=i))
    res = np.array(res)
    ok2 = bool((res[:, 0] <= 2.0 * res[:, 1] + 1e-3).all() and (res[:, 2] == 0).all())
    print(f'   {"PASS" if ok2 else "FAIL"}: max TV / MC-floor ratio {np.nanmax(res[:, 0] / np.maximum(res[:, 1], 1e-9)):.2f}, '
          f'max infeasible mass {res[:, 2].max():.0e}')
    assert ok2
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    test_ffbs()
