"""
Partial-lexicon HMM with UNK-order extension. JAX implementation.

Pure-functional model + Gibbs sampling, vmap-able across participants and
chains. State per trial t:
    B[t, s] in {-1 (UNK), 0..6}        belief for each of 7 words s
    O[t]   in {0..5, UNK_O=6}          word order

Transitions:
    B: at most one move per trial (identity / single-word / two-word swap)
       with log-costs (kappa, kappa_s); subject to partial-bijection
       constraint (no two words map to the same meaning).
    O: UNK can commit to any of the 6 orders at per-trial rate p_commit_o
       (uniform); committed orders switch among themselves at rate eps_o;
       once committed, no return to UNK.

Emission:
    committed order o:  role-position match given B (lapse mixture, rate lam)
    UNK order:          bag-of-words match (decoded set ⊆ candidate set)

The three model probabilities are each written once and shared by the
trajectory sampler and the likelihood:
    emit_p            choice probability (lapse mixture over surviving scenes)
    move_lp           belief-move log-probability (costs, normaliser Z)
    order_trans       order transition matrix

Public functions used by the sampler (sampler.py) and the simulator:
    traj_sweep        one Gibbs pass (order FFBS + per-word belief FFBS)
    stats_arrays      sufficient statistics of (B, O, data)
    loglik            log p(choices, B, O | params) from those statistics

All functions are pure JAX and jit/vmap-able. Per-participant data is fed
in via leading-axis batching; vmap externally across (chains, participants).

================================================================
Shape / meaning glossary used throughout this file
================================================================
Constants:
    N_W = 7    number of distinct word forms in the artificial lexicon
    N_M = 7    number of distinct meanings the words can map to
    N_O = 6    number of committed word orders (the 6 permutations of
               subject/verb/object): SVO, SOV, VSO, VOS, OSV, OVS
    UNK_O = 6  pseudo-7th order representing "not yet committed to any order"
    N_O_FULL = N_O + 1 = 7

Per-participant arrays (T = number of trials, usually 200):
    B  : int8  (T, N_W)
         B[t, s] is the participant's belief at trial t about which meaning
         word s refers to. -1 means UNK (no commitment yet).
    O  : int8  (T,)
         O[t] is the participant's word-order belief at trial t. Values
         0..N_O-1 are committed orders; UNK_O means uncommitted.
    U  : int32 (T, 3)
         U[t] are the three signal-word indices uttered on trial t (each in
         0..N_W-1; words may repeat within a signal).
    C  : int32 (T, 4, 3)
         C[t, j, r] is the meaning placed at role position r of candidate
         scene j on trial t. There are 4 candidate scenes per trial (4AFC).
    CH : int32 (T,)
         CH[t] in {0..3} is which candidate the participant picked.
    WO : int32 (N_O, 3)
         WO[o, p] is the ROLE (0 = agent/S, 1 = action/V, 2 = patient/O)
         expressed at position p of the 3-word signal under committed
         order o. E.g. for VOS, WO[o] = (1, 2, 0): position 0 carries
         the verb. NOTE the direction: position -> role, NOT role ->
         position; the signals in data_functions.py are built with the
         same convention. Used to map B-beliefs about the positional
         words to candidate-scene role slots in the committed-order
         emission rule.

Parameter vector (used everywhere as `params`):
    params : float32 (5,)
        params[0] = lam         lapse rate (probability of guessing uniformly
                                rather than using the model's prediction)
        params[1] = eps_o       per-trial probability of switching between
                                committed orders (small, ~chance of doubt)
        params[2] = kappa       log-cost of a single-word belief change
        params[3] = kappa_s     log-cost of a two-word swap belief change
                                (FIXED at KAPPA_S_FIXED, not fitted — see
                                to_nat and the README)
        params[4] = p_commit_o  per-trial probability that an UNK order
                                commits to one of the 6 committed orders
                                (uniform over the 6).
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.special import xlogy

UNK = -1            # belief value meaning "no commitment for this word"
UNK_O = 6           # order value meaning "no commitment to any word order"
N_W = 7             # number of distinct words in the lexicon
N_M = 7             # number of distinct meanings
N_O = 6             # number of committed word orders (permutations of S/V/O)
N_O_FULL = 7        # committed orders + UNK_O


# ---------- parameter transform ----------
# Transformed (sampling-scale) layout used by the hierarchical fit:
#     t[0] = logit(lam), t[1] = log(kappa), t[2] = logit(eps_o),
#     t[3] = logit(p_commit_o)
# Note the index permutation relative to the natural `params` order
# (lam, eps_o, kappa, kappa_s, p_commit_o). This function is the single
# source of truth for that mapping — do not re-implement it elsewhere.
#
# kappa_s is fixed, not fitted. In every posterior sample of every fit so
# far the sampled belief trajectories contained zero two-word swap
# transitions; with no swaps the likelihood is monotone increasing in
# kappa_s (only through the normaliser Z), so a fitted kappa_s runs off to
# wherever the hyperprior stops it. The swap transition type stays in the
# model (the reported fit was run with it); KAPPA_S_FIXED = 20 makes swap
# transitions ~e^-20 as likely as identity, so they are negligible.
KAPPA_S_FIXED = 20.0
N_TPAR = 4          # number of fitted transformed parameters

# Natural-scale layout consumed by the likelihood: params[i] for i in
NAT_NAMES = ('lam', 'eps_o', 'kappa', 'kappa_s', 'p_commit_o')
N_NAT = len(NAT_NAMES)
I_KAPPA_S = NAT_NAMES.index('kappa_s')
# Transformed coordinate k -> (natural slot, inverse link). This table is
# the mapping; to_nat (JAX, hot path) and tpar_to_nat_np (numpy, downstream
# summaries) are both generated from it, and tests/test_ffbs.py asserts they
# agree. config.PARAM_NAMES names the coordinates in the same order.
TPAR_SPEC = ((NAT_NAMES.index('lam'), 'logit'),         # t[0] = logit(lam)
             (NAT_NAMES.index('kappa'), 'log'),         # t[1] = log(kappa)
             (NAT_NAMES.index('eps_o'), 'logit'),       # t[2] = logit(eps_o)
             (NAT_NAMES.index('p_commit_o'), 'logit'))  # t[3] = logit(p_commit_o)
TPAR_NAT_INDEX = tuple(i for i, _ in TPAR_SPEC)      # natural slot of each t[k]
assert len(TPAR_SPEC) == N_TPAR
assert sorted(TPAR_NAT_INDEX + (I_KAPPA_S,)) == list(range(N_NAT))


def to_nat(t):
    """transformed (..., N_TPAR) -> natural (..., N_NAT) = NAT_NAMES order.

    kappa_s is filled in with the constant KAPPA_S_FIXED. JAX; used inside
    the jitted sampler."""
    cols = [None] * N_NAT
    for k, (i, link) in enumerate(TPAR_SPEC):
        cols[i] = jnp.exp(t[..., k]) if link == 'log' else jax.nn.sigmoid(t[..., k])
    cols[I_KAPPA_S] = jnp.full_like(cols[0], KAPPA_S_FIXED)
    return jnp.stack(cols, axis=-1)


def tpar_to_nat_np(t, k=None):
    """numpy (float64) inverse links for downstream summaries.

    t (..., N_TPAR) -> (..., N_TPAR) natural values IN TRANSFORMED-COORDINATE
    ORDER (lam, kappa, eps_o, p_commit_o), i.e. to_nat without the kappa_s
    slot and without the index permutation. With k given, t is a single
    coordinate k of any shape and its natural value is returned."""
    def inv(x, link):
        return np.exp(x) if link == 'log' else 1.0 / (1.0 + np.exp(-x))
    t = np.asarray(t, dtype=np.float64)
    if k is not None:
        return inv(t, TPAR_SPEC[k][1])
    return np.stack([inv(t[..., j], TPAR_SPEC[j][1]) for j in range(N_TPAR)], axis=-1)


# ---------- emission ----------

def _survivor_mask(o, b, u, cand, WO):
    """
    Which candidate scenes are compatible with the current belief?

    Single-trial helper. For a trial with order `o`, belief vector `b`,
    signal words `u`, candidate scenes `cand`, and the role permutation
    table `WO`, returns a length-4 boolean mask saying which candidates
    survive given the belief.

    Arguments
    ---------
    o    : int  ()                 order at this trial, 0..5 or UNK_O
    b    : int8 (N_W,)             belief vector at this trial (-1 = UNK)
    u    : int32 (3,)              the 3 signal words at this trial
    cand : int32 (4, 3)            the 4 candidate scenes (meaning per role)
    WO   : int32 (N_O, 3)          position -> role table per order

    Returns
    -------
    surv : bool (4,)               True iff candidate is consistent with b
    """

    # Belief about each of the 3 signal-position words. b[u] gathers the
    # belief entries for the three words actually uttered at this trial.
    # (3,) belief about signal words
    bu = b[u] 
    # Which of those three positions has a committed belief (not UNK).
    bu_committed = bu != -1 # (3,)

    # Committed-order survivor: role-position match:
    # Use a safe index for WO so vmap with o = UNK_O doesn't crash on OOB.
    o_safe = jnp.minimum(o, N_O - 1)
    # WO[o][p] = role expressed at signal position p under this order.
    # 0 = S, 1 = V, 2 = O
    roles_at_pos = WO[o_safe] # (3,)
    # cand_at_pos[j, p] = meaning candidate j assigns 
    # to the role carried by signal position p.
    # Directly comparable to bu[p].
    # (4, 3)
    cand_at_pos = cand[:, roles_at_pos]                  
    # A candidate scene is OK if, for every position whose word the participant
    # has a belief about, the candidate's meaning there matches the belief.
    # UNK positions are unconstrained (we OR with ~committed).
    ok_comm = (~bu_committed)[None, :] | (cand_at_pos == bu[None, :])
    # (4,)
    surv_comm = ok_comm.all(axis=1)                      

    # UNK word order survivor: decoded set is a subset of the candidate scene set
    # cand_set[j, m] = True iff meaning m appears anywhere in candidate j.
    # Built via scatter-True over the 3 meanings shown in each candidate.
    j_idx = jnp.arange(4)
    # (4, 7)
    cand_set = (
        jnp.zeros((4, N_M), dtype=jnp.bool_)
        .at[j_idx[:, None], cand].set(True)
    )                                                    
    # bu_safe: replace -1 entries with 0 so we never index cand_set at -1.
    # The corresponding result will be masked out below via bu_committed.
    bu_safe = jnp.where(bu_committed, bu, 0)
    # in_set[j, p] = is the believed meaning for signal-position p present
    # anywhere in candidate j? (Bag-of-words match, ignoring role.)
    # (4, 3)
    in_set = cand_set[:, bu_safe]                        
    # Unmasked positions contribute the actual in_set check; UNK positions
    # are vacuously True (no constraint).
    contains = jnp.where(bu_committed[None, :], in_set, True)
    surv_unk = contains.all(axis=1)                      # (4,)

    # Pick committed-order or UNK-order rule based on o.
    # (4,)
    return jnp.where(o == UNK_O, surv_unk, surv_comm)    


def emit_stat(o, b, u, cand, choice, WO):
    """
    The two numbers the emission probability depends on, besides lam:

    nS   : float ()   number of candidate scenes that survive the belief
    surv : float ()   1.0 iff the chosen scene survived

    "No information" (no committed belief about any heard word, or no
    surviving scene) is encoded as (nS, surv) = (4, 1), which makes emit_p
    a uniform guess, 1/4, whatever lam. Keeping these lam-free statistics
    apart from emit_p lets loglik() be evaluated for many parameter values
    without re-running the survivor rule.
    """
    surv = _survivor_mask(o, b, u, cand, WO)
    nS = surv.sum()
    no_info = (b[u] == UNK).all() | (nS == 0)
    return (jnp.where(no_info, 4.0, nS.astype(jnp.float32)),
            jnp.where(no_info, 1.0, surv[choice].astype(jnp.float32)))


def emit_p(nS, surv, lam):
    """P(choice) under the lapse mixture: with prob (1 - lam) a uniform pick
    among the nS surviving scenes, with prob lam a uniform guess among all 4."""
    return (1.0 - lam) * surv / nS + lam * 0.25


def emission_lp(o, b, u, cand, choice, lam, WO):
    """Single-trial log p(choice | o, b, lam). The correct scene is always
    candidate 0, so exp(emission_lp(..., choice=0, ...)) = P(correct)."""
    return jnp.log(emit_p(*emit_stat(o, b, u, cand, choice, WO), lam))


# ---------- belief transitions ----------

def _count_moves(b, u):
    """
    How many feasible belief-update moves does state (b, u) admit?

    Used as the normalizer of the transition distribution. We count two kinds:
        single-word change : reassign ONE signal-word's belief to UNK or
                             to a currently-free meaning (subject to the
                             partial-bijection constraint).
        two-word swap      : swap the beliefs of two distinct meanings
                             (at least one of which is heard in u).

    Arguments
    ---------
    b : int8 (N_W,)   belief vector
    u : int32 (3,)    signal words at this trial

    Returns
    -------
    n_single : int ()    number of feasible single-word moves
    n_swap   : int ()    number of feasible two-word swap moves
    """
    # present[m] = 1 iff meaning m is currently assigned to some word.
    # We zero out -1 entries in b (replacing them with index 0 but with a
    # zero add value) so scatter is well-defined.
    present = (
        jnp.zeros(N_M, dtype=jnp.int32)
        .at[jnp.where(b != -1, b, 0)].add(jnp.where(b != -1, 1, 0))
    )
    # nfree counts meanings not currently assigned to any word — these are
    # the meanings a word can be moved TO without breaking the bijection.
    nfree = (present == 0).sum()
    # is_first[i] = True iff signal-position i is the first occurrence of
    # its word (deduplicates: hearing the same word twice gives one move).
    is_first = jnp.array([
        ~jnp.any(jnp.array([u[i] == u[j] for j in range(i)]))
        for i in range(3)
    ])
    bu = b[u]
    bu_unk = bu == -1
    # contrib[i] = number of feasible moves for signal-position i.
    #   If currently UNK: can become any of `nfree` free meanings.
    #   Else (committed): can become UNK (1 option) OR any of nfree.
    contrib = jnp.where(bu_unk, nfree, 1 + nfree)
    # Only count each distinct word once.
    n_single = jnp.where(is_first, contrib, 0).sum()
    # ---- swap moves ----
    # heard[w] = True iff word w is uttered at this trial.
    heard = jnp.zeros(N_W, dtype=jnp.bool_).at[u].set(True)
    # A pair (i, j) is a feasible swap iff at least one of the two words
    # is heard AND their current beliefs differ.
    pair_heard = heard[:, None] | heard[None, :]
    pair_diff = b[:, None] != b[None, :]
    # Upper-triangular indices avoid counting (i, j) and (j, i) separately.
    iu, ju = jnp.triu_indices(N_W, k=1)
    n_swap = (pair_heard[iu, ju] & pair_diff[iu, ju]).sum()
    return n_single, n_swap


def _movetype(b_prev, b_cur, u):
    """Classify the move B[t-1] -> B[t].

    The model allows at most one move per trial, restricted to signal
    words. This function checks which of the four allowed move types the
    given transition is (so that _transition_lp can score it).

    Move-type codes:
        0  identity (no change)
        1  single-word change (the one changed word must be in u)
        2  two-word swap (at least one of the two changed words must be in
           u, and they must genuinely swap, i.e. b_cur[i] = b_prev[j] and
           vice versa)
       -1  infeasible (returned for any other change pattern)

    Arguments
    ---------
    b_prev : int8 (N_W,)    previous belief
    b_cur  : int8 (N_W,)    current belief
    u      : int32 (3,)     signal words at this trial

    Returns
    -------
    mt : int ()             move-type code (see above)
    """
    diff = b_prev != b_cur                               # (7,) which words changed
    ndiff = diff.sum()
    in_u = jnp.zeros(N_W, dtype=jnp.bool_).at[u].set(True)
    # single-word move: exactly one word changed AND that word is in u.
    single_ok = (ndiff == 1) & (in_u & diff).any()
    # swap move: exactly two words changed AND they swapped values AND
    # at least one of them is in u. (Matches _count_moves's swap-counting
    # rule, so that Z normalizes the transition distribution correctly.)
    # diff_idx[i] = i if word i changed, else -1. Then sort descending so
    # the (at most two) changed indices land in positions 0 and 1.
    diff_idx = jnp.where(diff, jnp.arange(N_W), -1)
    sorted_idx = jnp.sort(diff_idx, descending=True)
    d0, d1 = sorted_idx[0], sorted_idx[1]
    # Genuine swap: the two new values are each other's old values.
    swap_match = (b_cur[d0] == b_prev[d1]) & (b_cur[d1] == b_prev[d0])
    swap_in_u = in_u[d0] | in_u[d1]
    swap_ok = (ndiff == 2) & swap_match & swap_in_u
    return jnp.where(ndiff == 0, 0,
            jnp.where(single_ok, 1,
              jnp.where(swap_ok, 2, -1)))


def move_lp(mt, n1, n2, kappa, kappa_s):
    """Log-probability of a belief move of type mt (see _movetype) when n1
    single-word changes and n2 swaps are available (_count_moves):
    -cost - log Z, cost 0 / kappa / kappa_s for identity / single / swap,
    Z = 1 + n1 e^-kappa + n2 e^-kappa_s; -inf if infeasible (mt = -1)."""
    cost = jnp.where(mt == 0, 0.0, jnp.where(mt == 1, kappa, kappa_s))
    lp = -cost - jnp.log(1.0 + n1 * jnp.exp(-kappa) + n2 * jnp.exp(-kappa_s))
    return jnp.where(mt == -1, -jnp.inf, lp)


def _transition_lp(b_prev, b_cur, u, kappa, kappa_s):
    """Log P(B[t] = b_cur | B[t-1] = b_prev, u) under the cost model."""
    return move_lp(_movetype(b_prev, b_cur, u), *_count_moves(b_prev, u), kappa, kappa_s)


# ---------- forward-filter / backward-sample ----------

def ffbs(key, la0, trans_lp, emit_lp):
    """Exact draw of a discrete-state chain x_0..x_{T-1} given per-trial
    transition and emission log-probabilities, by forward filtering and
    backward sampling. Used for the order chain and for each word's belief
    chain. The state at t = 0 is given (one-hot la0) and never resampled.

    la0      (S,)          log-probabilities of x_0
    trans_lp (T-1, S, S)   trans_lp[t-1, i, j] = log p(x_t = j | x_{t-1} = i)
    emit_lp  (T-1, S)      emit_lp[t-1, j] = log p(data_t | x_t = j), t >= 1
    Returns x (T,) int32.
    """
    def fwd(la_prev, inp):                     # la_t = log p(x_t | data_1..t)
        tl, el = inp
        la = jax.nn.logsumexp(la_prev[:, None] + tl, axis=0) + el
        la = la - jax.nn.logsumexp(la)
        return la, la
    _, la_tail = lax.scan(fwd, la0, (trans_lp, emit_lp))
    las = jnp.concatenate([la0[None], la_tail], axis=0)             # (T, S)

    key, sub = random.split(key)
    x_last = random.categorical(sub, las[-1])

    def bwd(carry, inp):                       # x_t | x_{t+1} ~ la_t + trans
        key, x_next = carry
        la_t, tl_next = inp
        key, sub = random.split(key)
        x_t = random.categorical(sub, la_t + tl_next[:, x_next])
        return (key, x_t), x_t
    _, x_rev = lax.scan(bwd, (key, x_last), (las[:-1][::-1], trans_lp[::-1]))
    return jnp.concatenate([x_rev[::-1], x_last[None]])


# ---------- order chain ----------

def order_trans(eps_o, p_commit_o):
    """The (7, 7) order transition matrix M, M[i, j] = p(O[t] = j | O[t-1] = i).

    UNK stays UNK w.p. 1 - p_commit_o and otherwise commits to one of the 6
    orders uniformly; a committed order stays w.p. 1 - eps_o and otherwise
    switches to one of the other 5 uniformly; no return to UNK.
    """
    M = jnp.zeros((N_O_FULL, N_O_FULL))
    M = M.at[UNK_O, UNK_O].set(1.0 - p_commit_o)
    M = M.at[UNK_O, :N_O].set(p_commit_o / N_O)
    eye = jnp.eye(N_O)
    return M.at[:N_O, :N_O].set(eye * (1.0 - eps_o) + (1.0 - eye) * (eps_o / (N_O - 1)))


def ffbs_order(key, B, U, C, CH, WO, params):
    """Exact draw of O[0:T] from p(O | B, data, params): FFBS over the 7
    order states with O[0] = UNK_O. Returns O (T,) int8."""
    lam, eps_o, _, _, p_commit_o = params
    emit_lp = jax.vmap(lambda b, u, c, ch: jax.vmap(
        emission_lp, in_axes=(0, None, None, None, None, None, None))(
            jnp.arange(N_O_FULL), b, u, c, ch, lam, WO))(B[1:], U[1:], C[1:], CH[1:])
    log_M = jnp.log(order_trans(eps_o, p_commit_o))
    trans_lp = jnp.broadcast_to(log_M, (B.shape[0] - 1, N_O_FULL, N_O_FULL))
    la0 = jnp.where(jnp.arange(N_O_FULL) == UNK_O, 0.0, -jnp.inf)
    return ffbs(key, la0, trans_lp, emit_lp).astype(jnp.int8)


# ---------- belief chains: exact per-word FFBS ----------

N_BS = N_M + 1      # belief states per word: UNK plus the N_M meanings


def _word_transition_table(s, B, U, kappa, kappa_s):
    """Per-trial transition tables for word s's belief chain, other words
    frozen. Returns trans_lp (T-1, N_BS, N_BS) with
        trans_lp[t-1, xp, x] = log p(B[t, s] = x | B[t-1, s] = xp, B[:, w != s])
    where states xp, x index {UNK, 0..N_M-1} (value = state - 1).

    Equivalent to calling _transition_lp on all 64 candidate pairs (that
    is how it is verified: tests/test_ffbs.py, part 1), but
    ~17x cheaper. The move-type classification of _movetype is applied with
    the frozen words' change at trial t factored out:
      no frozen word changed : x == xp identity; else a single move by s,
                               legal iff s is heard at t.
      one frozen word w      : x == xp is a single move by w (legal iff w
                               heard); else two words changed, legal only
                               as a swap: x == old meaning of w, xp == new
                               meaning of w, and s or w heard.
      two frozen words       : only x == xp can be legal, and only if those
                               two words swapped with one of them heard.
      three or more          : infeasible.
    The normaliser Z (= _count_moves on the previous vector) depends on xp
    only, so it is computed once per xp instead of once per pair.
    """
    vals = jnp.arange(N_BS) - 1                                   # state -> value
    others = jnp.arange(N_W) != s
    Bp, Bn, Un = B[:-1], B[1:], U[1:]                             # (T-1, ...)
    # frozen words' change pattern at each transition
    diff_o = (Bp != Bn) & others[None, :]                         # (T-1, N_W)
    d = diff_o.sum(1)                                             # (T-1,)
    in_u = jax.vmap(lambda u: jnp.zeros(N_W, bool).at[u].set(True))(Un)
    s_in_u = in_u[:, s]

    def col(A, j):                                                # A[t, j[t]]
        return jnp.take_along_axis(A, j[:, None], axis=1)[:, 0]
    # d == 1: the one changed word w, its old / new meaning, heard?
    w = jnp.argmax(diff_o, axis=1)
    w_in_u, w_prev, w_cur = col(in_u, w), col(Bp, w), col(Bn, w)
    # d == 2: the two changed words (first and last set bit); a legal swap?
    i0 = jnp.argmax(diff_o, axis=1)
    i1 = N_W - 1 - jnp.argmax(diff_o[:, ::-1], axis=1)
    swap2 = (col(Bn, i0) == col(Bp, i1)) & (col(Bn, i1) == col(Bp, i0)) \
        & (col(in_u, i0) | col(in_u, i1))

    xp, x = vals[:, None], vals[None, :]                          # (8,1), (1,8)
    same = xp == x

    def movetype_t(d, s_in_u, w_in_u, swap2, w_prev, w_cur):
        mt0 = jnp.where(same, 0, jnp.where(s_in_u, 1, -1))
        swap_sw = (x == w_prev) & (xp == w_cur) & (s_in_u | w_in_u)
        mt1 = jnp.where(same, jnp.where(w_in_u, 1, -1),
                        jnp.where(swap_sw, 2, -1))
        mt2 = jnp.where(same, jnp.where(swap2, 2, -1), -1)
        return jnp.where(d == 0, mt0,
                         jnp.where(d == 1, mt1,
                                   jnp.where(d == 2, mt2, -1)))
    mt = jax.vmap(movetype_t)(d, s_in_u, w_in_u, swap2, w_prev, w_cur)   # (T-1, 8, 8)

    # normaliser per (t, xp): previous vector with slot s set to xp's value
    Bc_prev = jax.vmap(lambda b_t: jax.vmap(
        lambda v: b_t.at[s].set(v.astype(B.dtype)))(vals))(Bp)   # (T-1, 8, N_W)
    n1, n2 = jax.vmap(lambda bc, u: jax.vmap(
        lambda bp: _count_moves(bp, u))(bc))(Bc_prev, Un)         # (T-1, 8) each
    return move_lp(mt, n1[:, :, None], n2[:, :, None], kappa, kappa_s)


def ffbs_word(key, s, B, O, U, C, CH, WO, params):
    """Forward-filter / backward-sample word s's belief trajectory B[:, s]
    given the other words' trajectories, O and the data. Exact draw from
    p(B[:, s] | B[:, w != s], O, data, params).

    Why this is a plain HMM: freeze the other six words. The chain
    x_t = B[t, s] has N_BS = 8 states (UNK, meanings 0..N_M-1). The
    model's transition score at trial t is `_transition_lp` on the full
    belief vectors at t-1 and t; with the other words fixed that score is a
    function of (x_{t-1}, x_t) alone, so it is an 8 x 8 table per trial
    (built by _word_transition_table, which is equivalent to evaluating
    _transition_lp on all 64 candidate pairs). The emission at t depends on
    x_t alone and is evaluated by plugging every candidate state into the
    frozen vector and calling emission_lp.

    Two constraints are NOT part of `_transition_lp` and are imposed here
    as state masks:
      - partial bijection: meaning m is infeasible at trial t if some other
        word holds m at t (emission set to -inf);
      - the initial belief B[0] is fixed: the forward pass starts one-hot
        at the current B[0, s] and t = 0 is never resampled.

    Arguments
    ---------
    key    : PRNGKey
    s      : int ()                 word to resample (may be traced)
    B      : int8  (T, N_W)         current belief trajectory
    O      : int8  (T,)             current order trajectory
    U, C, CH, WO                    participant data, see glossary
    params : float (5,)             (lam, eps_o, kappa, kappa_s, p_commit_o)

    Returns
    -------
    B_new : int8 (T, N_W)   B with column s replaced by the fresh draw
    """
    lam, _, kappa, kappa_s, _ = params
    T = B.shape[0]
    vals = jnp.arange(N_BS, dtype=B.dtype) - 1                 # state -> value
    # Bc[t, x] = B[t] with slot s set to state x's value        (T, N_BS, N_W)
    Bc = jax.vmap(lambda b_t: jax.vmap(lambda v: b_t.at[s].set(v))(vals))(B)

    # Emission table (T, N_BS).
    def emit_t(o_t, bc_t, u_t, c_t, ch_t):
        return jax.vmap(emission_lp,
                        in_axes=(None, 0, None, None, None, None, None))(
            o_t, bc_t, u_t, c_t, ch_t, lam, WO)
    emit_lp = jax.vmap(emit_t)(O, Bc, U, C, CH)

    # Partial-bijection mask: meaning m is taken at t if another word holds it.
    others = jnp.arange(N_W) != s                                     # (N_W,)
    held = (B[:, :, None] == jnp.arange(N_M)[None, None, :]) \
        & others[None, :, None]                                       # (T, N_W, N_M)
    taken = held.any(axis=1)                                          # (T, N_M)
    feasible = jnp.concatenate([jnp.ones((T, 1), dtype=bool), ~taken],
                               axis=1)                                # (T, N_BS)
    emit_lp = jnp.where(feasible, emit_lp, -jnp.inf)

    # Transition tables (T-1, N_BS, N_BS):
    # trans_lp[t-1, xp, x] = log p(x_t = x | x_{t-1} = xp, other words).
    trans_lp = _word_transition_table(s, B, U, kappa, kappa_s)

    # t = 0 is pinned at the current state.
    x0 = B[0, s].astype(jnp.int32) + 1
    la0 = jnp.where(jnp.arange(N_BS) == x0, 0.0, -jnp.inf)
    x = ffbs(key, la0, trans_lp, emit_lp[1:])                         # (T,)
    return B.at[:, s].set(vals[x])


def ffbs_beliefs(key, B, O, U, C, CH, WO, params):
    """One systematic-scan Gibbs pass over the belief matrix: an exact
    per-word FFBS draw for each word s = 0..N_W-1 in turn, each conditional
    on the freshly updated other words. Returns (key, B_new)."""
    def body(s, carry):
        key, B = carry
        key, sub = random.split(key)
        return key, ffbs_word(sub, s, B, O, U, C, CH, WO, params)
    return lax.fori_loop(0, N_W, body, (key, B))


# ---------- one Gibbs sweep ----------

@partial(jax.jit, static_argnames=('n_ffbs',))
def traj_sweep(key, B, O, params, U, C, CH, WO, n_ffbs=1):
    """One full Gibbs pass over (O, B). Returns (key, B, O).

    Structure:
        1) Exact joint draw of O given B (forward-filter / backward-sample).
        2) n_ffbs passes of exact per-word FFBS over B (ffbs_beliefs):
           each word's whole trajectory is redrawn from its conditional
           given the other words and O.

    n_ffbs is declared static so the loop bound is concrete.

    Arguments
    ---------
    key                          : PRNGKey
    B, O                         : current state, see glossary
    params                       : (5,) parameter vector
    U, C, CH, WO                 : participant data, see glossary
    n_ffbs                       : int, per-word FFBS passes per sweep

    Returns
    -------
    key, B, O : updated key + post-sweep state
    """
    key, sub = random.split(key)
    # 1) Resample O exactly given B.
    O = ffbs_order(sub, B, U, C, CH, WO, params)

    # 2) n_ffbs exact per-word FFBS passes over B.
    def ffbs_step(i, carry):
        key, B = carry
        return ffbs_beliefs(key, B, O, U, C, CH, WO, params)
    key, B = lax.fori_loop(0, n_ffbs, ffbs_step, (key, B))
    return key, B, O


# ---------- sufficient statistics + log-likelihood ----------

@jax.jit
def stats_arrays(B, O, U, C, CH, WO):
    """Everything log p(choices, B, O | params) depends on except the
    parameters, so that loglik() can be evaluated for many parameter values
    per sweep without re-running the survivor and move rules (the sampler
    collapses them into counts, sampler.hist_stats, and evaluates the
    likelihood at every HMC leapfrog step).

    nS, surv  float (T,)     emission statistics per trial (emit_stat)
    mt        int   (T-1,)   belief move type per transition (_movetype)
    n1, n2    float (T-1,)   available single-word / swap moves (_count_moves)
    nO        float (7, 7)   order transition counts: nO[i, j] = #{t: O[t-1] = i, O[t] = j}
    """
    nS, surv = jax.vmap(emit_stat, in_axes=(0, 0, 0, 0, 0, None))(O, B, U, C, CH, WO)
    mt = jax.vmap(_movetype)(B[:-1], B[1:], U[1:])
    n1, n2 = jax.vmap(_count_moves)(B[:-1], U[1:])
    O32 = O.astype(jnp.int32)
    nO = jnp.zeros((N_O_FULL, N_O_FULL)).at[O32[:-1], O32[1:]].add(1.0)
    return nS, surv, mt, n1.astype(jnp.float32), n2.astype(jnp.float32), nO


@jax.jit
def loglik(params, nS, surv, mt, n1, n2, nO):
    """log p(choices, B, O | params) from stats_arrays' statistics, using the
    same emit_p, move_lp and order_trans as the FFBS steps."""
    lam, eps_o, kappa, kappa_s, p_commit_o = params
    # xlogy(n, p) = n log p with xlogy(0, 0) = 0: transitions that never
    # occur contribute nothing, including the zero entries of M and entries
    # a saturated sigmoid makes exactly 0 in float32 (0 * -inf would be nan)
    return (jnp.log(emit_p(nS, surv, lam)).sum()
            + move_lp(mt, n1, n2, kappa, kappa_s).sum()
            + xlogy(nO, order_trans(eps_o, p_commit_o)).sum())
