"""
Forward simulation from the partial-lexicon HMM (the generative model that
fit_hierarchical.py inverts). Used for parameter recovery (param_recovery.py).

Two ways to obtain the latent trajectories (B, O) of a synthetic participant:

  simulate_latents(rng, tpar, U)          — draw B and O forward from the
        transition model (belief random walk with costs kappa / kappa_s,
        order commit / switch with p_commit_o / eps_o), starting all-UNK.
        This is the model's own prior over trajectories. NOTE that it is a
        *measurement* model: beliefs are not pulled toward the true lexicon,
        so a prior-simulated participant holds a random (mostly wrong)
        lexicon and their accuracy against the true scene stays near chance.
        The inference problem (which beliefs generate these choices, and how
        often do they move) is the same as on real data; only the "accuracy
        vs truth" summaries are not comparable.

  posterior (B, O) from a fit                — a joint posterior draw of the
        real data gives learning-like trajectories (lexicon converging to the
        truth, order committing to the true one); param_recovery.py's
        `posterior` mode regenerates choices from such a draw.

simulate_choices(rng, B, O, lam, U, C)     — choices from the emission model
        given trajectories: with prob lam a uniform guess, otherwise uniform
        over the candidate scenes that survive the belief (model._survivor_mask;
        no survivors or no committed heard word -> uniform), exactly the
        distribution emission_lp / emit_p_correct evaluate.

All functions work on the real stimuli (signals U, candidate scenes C) of the
325 participants, so a simulated dataset is the real experiment with the
choices replaced. data.py loads such a dataset when EA_SIM_DATA points to an
.npz written by save_sim_dataset.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

import model as NB

N_W, N_M, N_O, UNK_O = NB.N_W, NB.N_M, NB.N_O, NB.UNK_O


# ------------------------------------------------------------ latents ----

def _feasible_moves(b, u):
    """All feasible belief updates from b given heard words u, as a list of
    (new_b, cost_code) with cost_code 0 = identity, 1 = single (cost kappa),
    2 = swap (cost kappa_s). Enumerates exactly the move set counted by
    model._count_moves and scored by model._transition_lp."""
    moves = [(b.copy(), 0)]
    present = np.zeros(N_M, bool)
    present[b[b != -1]] = True
    free = np.flatnonzero(~present)
    for w in dict.fromkeys(int(x) for x in u):          # distinct heard words
        if b[w] != -1:                                  # committed -> UNK
            nb = b.copy(); nb[w] = -1; moves.append((nb, 1))
        for m in free:                                  # -> any free meaning
            nb = b.copy(); nb[w] = m; moves.append((nb, 1))
    heard = np.zeros(N_W, bool)
    heard[u] = True
    for i in range(N_W):
        for j in range(i + 1, N_W):
            if (heard[i] or heard[j]) and b[i] != b[j]:
                nb = b.copy(); nb[i], nb[j] = b[j], b[i]; moves.append((nb, 2))
    return moves


def simulate_latents(rng, tpar, U, kappa_s=NB.KAPPA_S_FIXED):
    """Draw (B, O) for one participant from the transition model.

    rng   : np.random.Generator
    tpar  : (N_TPAR,) transformed params (config.PARAM_NAMES order)
    U     : (T, 3) int signal words per trial

    Returns B (T, N_W) int8 (-1 = UNK), O (T,) int8 (UNK_O = uncommitted).
    B[0] is all-UNK and O[0] = UNK_O, as the sampler assumes.
    """
    lam, kappa, eps_o, p_commit = NB.tpar_to_nat_np(tpar)   # transformed-coord order
    T = U.shape[0]
    B = np.full((T, N_W), -1, np.int8)
    O = np.full(T, UNK_O, np.int8)
    cost = np.array([0.0, kappa, kappa_s])
    for t in range(1, T):
        moves = _feasible_moves(B[t - 1], U[t])
        w = np.exp(-cost[[c for _, c in moves]])
        B[t] = moves[rng.choice(len(moves), p=w / w.sum())][0]
        o = O[t - 1]
        if o == UNK_O:
            O[t] = rng.integers(N_O) if rng.random() < p_commit else UNK_O
        else:
            if rng.random() < eps_o:
                O[t] = (o + 1 + rng.integers(N_O - 1)) % N_O
            else:
                O[t] = o
    return B, O


# ------------------------------------------------------------ choices ----

@jax.jit
def choice_probs(B, O, lam, U, C, WO):
    """Per (participant, trial) probability of each of the 4 candidates.
    B (P, T, 7), O (P, T), lam (P,), U (P, T, 3), C (P, T, 4, 3) -> (P, T, 4).
    Same mixture as model.emission_lp, for all four choices at once."""
    def one(o, b, u, c, lam_p):
        surv = NB._survivor_mask(o, b, u, c, WO)
        nS = surv.sum()
        no_info = ~(b[u] != -1).any() | (nS == 0)
        p = (1.0 - lam_p) * surv.astype(jnp.float32) / jnp.maximum(nS, 1) + lam_p * 0.25
        return jnp.where(no_info, 0.25, p)
    per_ppt = vmap(one, in_axes=(0, 0, 0, 0, None))
    return vmap(per_ppt)(O, B, U, C, lam)


def simulate_choices(rng, B, O, lam, U, C, WO):
    """Draw one choice sequence per participant from the emission model.
    B (P, T, 7), O (P, T), lam (P,), U (P, T, 3), C (P, T, 4, 3).
    Returns CH (P, T) int8 in 0..3 (0 = the correct scene)."""
    p = np.asarray(choice_probs(jnp.asarray(B), jnp.asarray(O),
                                jnp.asarray(lam, jnp.float32),
                                jnp.asarray(U), jnp.asarray(C), jnp.asarray(WO)))
    u = rng.random(p.shape[:2])[..., None]
    return (u > np.cumsum(p, axis=-1)).sum(-1).clip(0, 3).astype(np.int8)


# ------------------------------------------------------------ dataset ----

def save_sim_dataset(path, choices_TP, truth):
    """Write a simulated dataset for data.py (EA_SIM_DATA). choices_TP is
    (T, P) with 0 = correct, as data.choices; `truth` is a dict of arrays
    (B, O, tpar, hyper components, ...) stored alongside for the recovery
    analysis."""
    np.savez_compressed(path, choices=np.asarray(choices_TP, np.int8),
                        **{f'true_{k}': np.asarray(v) for k, v in truth.items()})
    return path
