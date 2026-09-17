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

Public functions used by fit_hierarchical.py and infer_per_participant.py:
    traj_sweep        one Gibbs pass (order FFBS + per-word belief FFBS
                      + n_swap swap MH moves; optional legacy block MH)
    stats_arrays      per-trial sufficient statistics
    loglik            data log-likelihood given suff. stats and params

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
    HT : int32 (N_W, T)  ("history-of-trials-by-word", padded with -1)
         HT[s, :HC[s]] are the trial indices on which word s was uttered.
    HC : int32 (N_W,)
         HC[s] is the number of trials on which word s was uttered.

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
# wherever the hyperprior stops it. The swap transition type is kept so
# that swap_move (a relabeling MH proposal that escapes lexicon-permutation
# modes) remains admissible; KAPPA_S_FIXED = 20 makes swap transitions
# ~e^-20 as likely as identity, i.e. only accepted on overwhelming evidence.
KAPPA_S_FIXED = 20.0
N_TPAR = 4          # number of fitted transformed parameters

# Natural-scale layout consumed by the likelihood: params[i] for i in
NAT_NAMES = ('lam', 'eps_o', 'kappa', 'kappa_s', 'p_commit_o')
N_NAT = len(NAT_NAMES)
I_KAPPA_S = NAT_NAMES.index('kappa_s')
# Transformed coordinate k -> (natural slot, inverse link). This table is
# the mapping; to_nat (JAX, hot path) and tpar_to_nat_np (numpy, downstream
# summaries) are both generated from it, and check_ffbs.py asserts they
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


def _surv_stats(o, b, u, cand, choice, WO):
    """
    Shared per-trial survivor statistics for the three emission views.

    "No information" happens if the participant has no committed beliefs
    about the uttered words OR if no candidate is compatible (degenerate
    state). Either way the emission collapses to a uniform-over-4 guess.

    Arguments
    ---------
    o      : int  ()                order at this trial
    b      : int8 (N_W,)            belief vector at this trial, word->meaning
    u      : int32 (3,)             signal words at this trial
    cand   : int32 (4, 3)           candidate scenes
    choice : int32 ()               which candidate (0..3) to evaluate
    WO     : int32 (N_O, 3)         position -> role table

    Returns
    -------
    nS          : int   ()    number of surviving candidates
    surv_choice : float ()    1.0 iff `choice` survived
    no_info     : bool  ()    collapse-to-uniform flag (see above)
    """
    # meaning of each signal word
    bu = b[u]
    # any of the signal words have a committed meaning?
    any_comm = (bu != -1).any()
    # which candidate scenes survive?
    surv = _survivor_mask(o, b, u, cand, WO)
    # number of surviving candidates
    nS = surv.sum()
    # did the chosen candidate survive?
    surv_choice = surv[choice].astype(jnp.float32)
    # no info if no committed beliefs or no surviving candidates
    no_info = ~any_comm | (nS == 0)
    return nS, surv_choice, no_info


def emission_lp(o, b, u, cand, choice, lam, WO):
    """
    Single-trial log p(choice | o, b, lam, ...) under the lapse mixture:
    with prob (1 - lam) pick uniformly among the candidates that survive
    the belief, with prob lam guess uniformly among all 4 candidates.
    """
    nS, surv_choice, no_info = _surv_stats(o, b, u, cand, choice, WO)
    p = (1.0 - lam) * surv_choice / jnp.maximum(nS, 1) + lam * 0.25
    return jnp.where(no_info, jnp.log(0.25), jnp.log(p))


def emit_p_correct(o, b, u, cand, lam, WO):
    """
    Per-trial Pr(participant picks the CORRECT candidate). The data is
    laid out so that the correct candidate is always at index 0, so this
    is emission_lp's mixture evaluated at choice=0, on the prob scale.
    """
    nS, surv0, no_info = _surv_stats(o, b, u, cand, 0, WO)
    return jnp.where(
        no_info, 0.25,
        (1.0 - lam) * surv0 / jnp.maximum(nS, 1) + lam * 0.25)


def emit_stat(o, b, u, cand, choice, WO):
    """
    Per-trial emission sufficient statistics (without lam), consumed by
    stats_arrays so that loglik() can be evaluated for many parameter
    settings without re-running the survivor logic.

    Convention: in the "no information" case we return (nS=4, surv=1) so
    that the lapse formula in loglik() degenerates exactly to 1/4.

    Arguments
    ---------
    o : int ()                order at this trial
    b : int8 (N_W,)           belief vector at this trial
    u : int32 (3,)            signal words at this trial
    cand : int32 (4, 3)       candidate scenes
    choice : int32 ()         which candidate (0..3) to evaluate
    WO : int32 (N_O, 3)       position -> role table

    Returns
    -------
    nS          : float ()    number of surviving candidates (or 4)
    surv_choice : float ()    1.0 if the chosen candidate survived (or 1)
    """
    nS, surv_choice, no_info = _surv_stats(o, b, u, cand, choice, WO)
    return (jnp.where(no_info, 4.0, nS.astype(jnp.float32)),
            jnp.where(no_info, 1.0, surv_choice))


# ---------- belief transitions ----------

def _count_moves(b, u):
    """
    How many feasible belief-update moves does state (b, u) admit?

    Used both as the proposal-set size in MH and as the normalizer of the
    transition distribution. We count two kinds:
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


def _transition_lp(b_prev, b_cur, u, kappa, kappa_s):
    """Log P(B[t] = b_cur | B[t-1] = b_prev, u) under the cost model.

    Move costs: identity = 0, single-word = kappa, two-word swap = kappa_s.
    The trial's transition distribution is normalized by Z = sum of
    exp(-cost) over all feasible moves (identity + n_single singles +
    n_swap swaps).

    Arguments
    ---------
    b_prev, b_cur, u   : see _movetype
    kappa              : float ()   log-cost of a single-word change
    kappa_s            : float ()   log-cost of a two-word swap

    Returns
    -------
    lp : float ()      log-probability of the transition (-inf if infeasible)
    """
    mt = _movetype(b_prev, b_cur, u)
    n_single, n_swap = _count_moves(b_prev, u)
    # Normalizer over all feasible moves available from (b_prev, u).
    Z = 1.0 + n_single * jnp.exp(-kappa) + n_swap * jnp.exp(-kappa_s)
    cost = jnp.where(mt == 0, 0.0,
            jnp.where(mt == 1, kappa,
              jnp.where(mt == 2, kappa_s, jnp.inf)))
    lp = -cost - jnp.log(Z)
    return jnp.where(mt == -1, -jnp.inf, lp)


# ---------- Forward Filtering Backward Ordering over order ----------

def _build_order_transition(eps_o, p_commit_o):
    """Build the (7, 7) order transition matrix M.

    Row i = O[t-1], column j = O[t]. The committed orders form an
    irreducible block where each order stays with p (1-eps_o) and jumps to
    each of the other 5 with p eps_o/5. The UNK row commits to one of the
    6 orders w.p. p_commit_o (uniform) and stays UNK otherwise.

    Note: there is no return-to-UNK from a committed order — once the
    participant commits, they don't un-commit.

    Arguments
    ---------
    eps_o       : float ()   per-trial committed-order switch rate
    p_commit_o  : float ()   per-trial commit-from-UNK rate

    Returns
    -------
    M : float (N_O_FULL, N_O_FULL)   row-stochastic transition matrix
    """
    M = jnp.zeros((N_O_FULL, N_O_FULL))
    # UNK row: stay UNK with prob (1 - p_commit_o), else commit uniformly.
    M = M.at[UNK_O, UNK_O].set(1.0 - p_commit_o)
    M = M.at[UNK_O, :N_O].set(p_commit_o / N_O)
    # Committed block: identity-on-diag at (1-eps_o), uniform off-diag.
    eye = jnp.eye(N_O)
    comm = eye * (1.0 - eps_o) + (1.0 - eye) * (eps_o / (N_O - 1))
    M = M.at[:N_O, :N_O].set(comm)
    return M


def ffbs_order(key, B, U, C, CH, WO, params):
    """Forward-filter / backward-sample new O given B and data.

    Standard FFBS over the 7-state HMM with emission p(choice | o, B[t])
    and transition matrix from _build_order_transition. Conditional on B,
    this gives an exact joint draw from p(O[0:T] | B, data, params).

    Arguments
    ---------
    key    : PRNGKey
    B      : int8  (T, N_W)         current belief trajectory
    U      : int32 (T, 3)           signal words per trial
    C      : int32 (T, 4, 3)        candidate scenes per trial
    CH     : int32 (T,)             chosen candidate per trial
    WO     : int32 (N_O, 3)         position -> role table
    params : float (5,)             (lam, eps_o, kappa, kappa_s, p_commit_o)

    Returns
    -------
    O : int8 (T,)   freshly sampled order trajectory
    """
    lam, eps_o, _, _, p_commit_o = params

    # ---- emission matrix: log p(choice_t | o, B[t]) for each (t, o) ----
    # Then exponentiate after subtracting per-row max to prevent under/overflow
    # in the forward sums below; the per-row normalization cancels out anyway.
    def per_t(b_t, u_t, c_t, ch_t):
        # vmap over the 7 possible orders for a single trial.
        return jax.vmap(emission_lp, in_axes=(0, None, None, None, None, None, None))(
            jnp.arange(N_O_FULL), b_t, u_t, c_t, ch_t, lam, WO)
    # Outer vmap over the T trials.
    log_emit = jax.vmap(per_t)(B, U, C, CH)              # (T, 7)
    emit = jnp.exp(log_emit - log_emit.max(axis=1, keepdims=True))

    trans = _build_order_transition(eps_o, p_commit_o)

    # ---- initial filtered distribution alpha[0] ----
    # Model assumes everyone starts UNK at t=0 (deterministic), so the
    # only nonzero entry is at UNK_O — weighted by its emission at t=0.
    alpha0 = jnp.zeros(N_O_FULL).at[UNK_O].set(emit[0, UNK_O])
    alpha0 = alpha0 / alpha0.sum()

    # ---- forward filtering ----
    # alpha_t ∝ (alpha_{t-1} @ trans) * emit[t], normalized.
    def fwd(alpha_prev, emit_t):
        a = (alpha_prev @ trans) * emit_t
        a = a / jnp.maximum(a.sum(), 1e-30)
        return a, a
    _, alphas_tail = lax.scan(fwd, alpha0, emit[1:])     # (T-1, 7)
    # Glue alpha[0] back on the front to get the full (T, 7) array.
    alphas = jnp.concatenate([alpha0[None], alphas_tail], axis=0)

    # ---- backward sampling ----
    # Sample O[T-1] from alpha[T-1], then walk backwards drawing each
    # O[t] from p(O[t] | O[t+1], data) ∝ alpha[t] * trans[:, O[t+1]].
    key, sub = random.split(key)
    O_T = random.categorical(sub, jnp.log(alphas[-1] + 1e-30))

    def bwd(carry, alpha_t):
        key, O_next = carry
        key, sub = random.split(key)
        w = alpha_t * trans[:, O_next]
        O_t = random.categorical(sub, jnp.log(w + 1e-30))
        return (key, O_t), O_t
    # Scan backwards by reversing the alpha array, then un-reverse.
    (_, _), O_rev = lax.scan(bwd, (key, O_T), alphas[:-1][::-1])
    return jnp.concatenate([O_rev[::-1], O_T[None]], axis=0).astype(jnp.int8)


# ---------- block + swap MH over B ----------

# ---------- exact per-word belief FFBS ----------

N_BS = N_M + 1      # belief states per word: UNK plus the N_M meanings


def _word_transition_table(s, B, U, kappa, kappa_s):
    """Per-trial transition tables for word s's belief chain, other words
    frozen. Returns trans_lp (T-1, N_BS, N_BS) with
        trans_lp[t-1, xp, x] = log p(B[t, s] = x | B[t-1, s] = xp, B[:, w != s])
    where states xp, x index {UNK, 0..N_M-1} (value = state - 1).

    Equivalent to calling _transition_lp on all 64 candidate pairs (that
    is how it is verified: check_ffbs.py, part 1), but
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
    logZ = jnp.log(1.0 + n1 * jnp.exp(-kappa) + n2 * jnp.exp(-kappa_s))
    cost = jnp.where(mt == 0, 0.0,
                     jnp.where(mt == 1, kappa,
                               jnp.where(mt == 2, kappa_s, jnp.inf)))
    return jnp.where(mt == -1, -jnp.inf, -cost - logZ[:, :, None])


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
    as state masks, matching what block_move / swap_move enforced through
    their proposals:
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

    # Forward filter in log space, t = 0 pinned at the current state.
    x0 = B[0, s].astype(jnp.int32) + 1
    la0 = jnp.where(jnp.arange(N_BS) == x0, 0.0, -jnp.inf)

    def fwd(la_prev, inp):
        tl, el = inp
        la = jax.nn.logsumexp(la_prev[:, None] + tl, axis=0) + el
        la = la - jax.nn.logsumexp(la)
        return la, la
    _, la_tail = lax.scan(fwd, la0, (trans_lp, emit_lp[1:]))
    las = jnp.concatenate([la0[None], la_tail], axis=0)             # (T, N_BS)

    # Backward sample.
    key, sub = random.split(key)
    x_last = random.categorical(sub, las[-1])

    def bwd(carry, inp):
        key, x_next = carry
        la_t, tl_next = inp
        key, sub = random.split(key)
        x_t = random.categorical(sub, la_t + tl_next[:, x_next])
        return (key, x_t), x_t
    (_, _), x_rev = lax.scan(bwd, (key, x_last),
                             (las[:-1][::-1], trans_lp[::-1]))
    x = jnp.concatenate([x_rev[::-1], x_last[None]])                # (T,)
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


def _partial_lp(O, B, U, C, CH, WO, params, a, e):
    """log p(data + transitions) over a window of trials, JAX-friendly.

    Computes the sum of emission log-probs on [a, e) plus belief-transition
    log-probs on (a, e]. Used as the un-normalized log-posterior in the MH
    accept/reject for block_move and swap_move (where everything outside
    the window cancels in the ratio).

    Implementation: rather than slicing (which would break under jit with
    dynamic a, e), we evaluate emission and transition log-probs at every
    trial 0..T-1 and zero out those outside the window via a mask.

    Arguments
    ---------
    O, B, U, C, CH, WO, params : see ffbs_order
    a, e : int ()    window of trials [a, e) (e is exclusive for emission;
                     transition is summed over t in [max(a,1), min(e, T-1)])

    Returns
    -------
    lp : float ()    sum of in-window emission + transition log-probs
    """
    lam, _, kappa, kappa_s, _ = params
    T = B.shape[0]
    t_idx = jnp.arange(T)
    # ---- emission term ----
    # vmap emission_lp over all T trials; mask down to [a, e).
    emit_lp = jax.vmap(emission_lp, in_axes=(0, 0, 0, 0, 0, None, None))(
        O, B, U, C, CH, lam, WO)
    emit_mask = (t_idx >= a) & (t_idx < e)
    emit_sum = jnp.where(emit_mask, emit_lp, 0.0).sum()
    # ---- transition term ----
    # For each t in 1..T-1 compute log P(B[t] | B[t-1], U[t]).
    def trans_at(t):
        return _transition_lp(B[t - 1], B[t], U[t], kappa, kappa_s)
    trans_lp = jax.vmap(trans_at)(jnp.arange(1, T))      # (T-1,)
    # We want transitions touching the window — i.e. those whose t is in
    # [max(a,1), min(e, T-1)] (clamped on both ends to stay in-range).
    trans_mask = (jnp.arange(1, T) >= jnp.maximum(a, 1)) & \
                 (jnp.arange(1, T) <= jnp.minimum(e, T - 1))
    trans_sum = jnp.where(trans_mask, trans_lp, 0.0).sum()
    return emit_sum + trans_sum


def block_move(key, B, O, U, C, CH, WO, params, HT, HC):
    """MH proposal: change one word's belief over a contiguous block.

    Picks a word s, picks an interval [a, e) bounded by trials on which s
    appears, then proposes a single new constant value for B[a:e, s].
    Accept/reject via standard MH against _partial_lp on the window.

    Arguments
    ---------
    key                                  : PRNGKey
    B, O, U, C, CH, WO, params, HT, HC   : see module-level glossary

    Returns
    -------
    B_new : int8 (T, N_W)   updated belief (unchanged if proposal rejected)

    Proposal symmetry
    -----------------
    The accept ratio uses the simple form log_u < new_lp - old_lp, which is
    valid only if the proposal density is symmetric: q(B_new | B) = q(B | B_new).
    Here that holds because every step of the proposal chain is either
    data-only (so independent of B) or has the same cardinality from either
    state:
      - pick word s            : uniform over N_W, data-free        ✓
      - pick start a           : uniform over HC[s] options, data-only ✓
      - pick end e             : uniform over (n_later + 1) options,
                                 derived from HT[s] only            ✓
      - pick new value v_new   : uniform over `cand_free`. Its size K
                                 depends on the "other-held" set H of
                                 meanings that some OTHER word w != s
                                 holds anywhere in the block. H does NOT
                                 depend on B[a:e, s] (we explicitly mask
                                 out word s in `free_for`), so H is the
                                 same in forward and reverse. The current
                                 value v is excluded from cand_free in
                                 both directions, giving |cand_free| = K
                                 in both — so q(v_new | v_old) =
                                 q(v_old | v_new) = 1/K.                ✓
    Validity of the (a, e) window is also direction-invariant: a and e are
    always trials where word s appears (or e = T), so the boundary
    transitions B[a-1] -> B[a] and B[e-1] -> B[e] are single-word moves
    in BOTH directions (or absent when e = T).
    """
    T = B.shape[0]
    # 5 sub-keys: word, block-start, block-end, new-value, MH-uniform.
    keys = random.split(key, 5)
    # 1) Pick a word s uniformly.
    s = random.randint(keys[0], (), 0, N_W)
    cnt = HC[s]
    # If this word never appeared, we cannot propose anything — mark invalid.
    valid = cnt > 0
    # 2) Pick a start trial a uniformly from the trials where s appears.
    a_idx = jnp.where(valid, random.randint(keys[1], (), 0, jnp.maximum(cnt, 1)), 0)
    a = HT[s, a_idx]
    # We never touch t=0 (the initial belief is fixed to all-UNK).
    valid = valid & (a >= 1)
    # 3) Pick an end trial e. Candidates are trials with word s strictly
    #    after a, PLUS the past-the-end sentinel T.
    later = (HT[s] > a) & (jnp.arange(T) < cnt)
    n_later = later.sum()
    r = random.randint(keys[2], (), 0, n_later + 1)
    # If r selects the sentinel, set e = T; else pick the r-th later trial.
    later_positions = jnp.where(later, HT[s], T + 1)
    sorted_later = jnp.sort(later_positions)
    e = jnp.where(r == n_later, T, sorted_later[r])

    # The current value v held across the block. It must be the same on
    # every trial in [a, e) — if not, we treat the proposal as invalid.
    v = B[a, s]
    t_idx = jnp.arange(T)
    in_block = (t_idx >= a) & (t_idx < e)
    constant_in_block = jnp.all(jnp.where(in_block, B[:, s] == v, True))
    valid = valid & constant_in_block

    # 4) Enumerate feasible new values for word s on this block. A value m
    #    is "free" if no OTHER word holds m at any trial inside the block.
    def free_for(mm):
        # held[t, w] = True iff word w (other than s) holds meaning mm at t.
        held = (B == mm) & (jnp.arange(N_W)[None, :] != s)
        held_any_t = jnp.any(jnp.where(in_block[:, None], held, False))
        return ~held_any_t
    free_mask = jax.vmap(free_for)(jnp.arange(N_M))      # (7,)
    # Build the 8 candidate values: UNK plus the 7 meanings, with -1 first.
    cand_vals = jnp.concatenate([jnp.array([-1]), jnp.arange(N_M)])
    cand_free = jnp.concatenate([
        # Going back to UNK is only allowed when v is currently committed.
        jnp.array([v != -1]),
        # Otherwise any free meaning that isn't already v.
        free_mask & (jnp.arange(N_M) != v),
    ])
    nf = cand_free.sum()
    valid = valid & (nf > 0)
    # 5) Sample new value uniformly among the feasible ones.
    weights = cand_free.astype(jnp.float32)
    v_new = random.categorical(keys[3], jnp.log(weights + 1e-30))
    v_new = cand_vals[v_new]

    # 6) Build the proposed B and MH-accept/reject by partial log-prob.
    B_new = jnp.where(in_block[:, None] & (jnp.arange(N_W)[None, :] == s),
                      v_new, B).astype(B.dtype)
    old_lp = _partial_lp(O, B,     U, C, CH, WO, params, a, e)
    new_lp = _partial_lp(O, B_new, U, C, CH, WO, params, a, e)
    log_u = jnp.log(random.uniform(keys[4]))
    accept = valid & (log_u < (new_lp - old_lp))
    return jnp.where(accept, B_new, B).astype(B.dtype)


def swap_move(key, B, O, U, C, CH, WO, params):
    """
    MH proposal: swap the beliefs of two words from some trial onwards.

    Picks two distinct words (w1, w2) and a trial t0 (uniform among trials
    ≥ 1 where at least one of them is uttered), then swaps B[t0:, w1] and
    B[t0:, w2]. Helps the sampler escape lexicon-order confounds where the
    only difference between two posterior modes is a permutation of two
    word-meaning mappings.

    Arguments
    ---------
    key                            : PRNGKey
    B, O, U, C, CH, WO, params     : see module-level glossary

    Returns
    -------
    B_new : int8 (T, N_W)   updated belief (unchanged if proposal rejected)

    Proposal symmetry
    -----------------
    The swap is its own inverse: applying the same (w1, w2, t0) to B_new
    recovers B. The picks (w1, w2) are uniform and data-free. The set of
    eligible t0 values is {t >= 1 : w1 in U[t] OR w2 in U[t]}, a function
    of U only — same in forward and reverse. The "non-trivial" validity
    check (B[:, w1] != B[:, w2] somewhere in the tail) is also direction-
    invariant since the swap just relabels those two columns. So
    q(B_new | B) = q(B | B_new) and the simple log_u < new_lp - old_lp
    acceptance is correct.
    """
    T = B.shape[0]
    # 5 sub-keys: w1, w2, t0-pick, MH-uniform (keys[3]); keys[4] is unused
    # (kept so the key stream of existing checks is unchanged).
    keys = random.split(key, 5)
    w1 = random.randint(keys[0], (), 0, N_W)
    w2 = random.randint(keys[1], (), 0, N_W)
    valid = w1 != w2
    # Eligible trials: t >= 1 AND at least one of {w1, w2} appears in U[t].
    eligible = jnp.any((U == w1) | (U == w2), axis=1) & (jnp.arange(T) >= 1)
    n_elig = eligible.sum()
    valid = valid & (n_elig > 0)
    r = random.randint(keys[2], (), 0, jnp.maximum(n_elig, 1))
    # Pick the r-th eligible trial. T+1 sentinel pushes non-eligible
    # entries to the right after sorting.
    elig_positions = jnp.where(eligible, jnp.arange(T), T + 1)
    sorted_pos = jnp.sort(elig_positions)
    t0 = sorted_pos[r]
    # If B[w1] == B[w2] throughout the tail, the swap is a no-op; skip.
    in_tail = jnp.arange(T) >= t0
    diff = (B[:, w1] != B[:, w2]) & in_tail
    valid = valid & diff.any()
    # Build the proposed B by swapping columns w1 and w2 on the tail only.
    tmp1 = jnp.where(in_tail, B[:, w2], B[:, w1]).astype(B.dtype)
    tmp2 = jnp.where(in_tail, B[:, w1], B[:, w2]).astype(B.dtype)
    B_new = B.at[:, w1].set(tmp1).at[:, w2].set(tmp2)
    # MH on the affected window [t0, T).
    old_lp = _partial_lp(O, B,     U, C, CH, WO, params, t0, T)
    new_lp = _partial_lp(O, B_new, U, C, CH, WO, params, t0, T)
    log_u = jnp.log(random.uniform(keys[3]))
    accept = valid & (log_u < (new_lp - old_lp))
    return jnp.where(accept, B_new, B).astype(B.dtype)


# ---------- one Gibbs sweep ----------

@partial(jax.jit, static_argnames=('n_block', 'n_swap', 'n_ffbs'))
def traj_sweep(key, B, O, params, U, C, CH, WO, HT, HC,
               n_block=0, n_swap=0, n_ffbs=1):
    """One full Gibbs pass over (O, B). Returns (key, B, O).

    Structure:
        1) Exact joint draw of O given B (forward-filter / backward-sample).
        2) n_ffbs passes of exact per-word FFBS over B (ffbs_beliefs):
           each word's whole trajectory is redrawn from its conditional
           given the other words. This is the workhorse for B.
        3) n_block MH proposals for contiguous block changes of B
           (block_move). Legacy kernel, superseded by 2): blind proposals
           with ~0.3% acceptance on real data. Off by default; kept for
           reference and for old-vs-new checks.
        4) n_swap MH proposals for two-word lexicon swaps on a tail of B
           (swap_move). Off by default: per-word FFBS reaches a swapped
           labelling in two exact draws via the UNK state, and on the real
           data FFBS-only and FFBS+swap chains give the same marginals and
           log-likelihoods within seed-to-seed noise, while the swap stage
           (~0.1% acceptance) costs ~40% of the sweep. Kept as an option.

    All B stages share O for the whole sweep. n_ffbs, n_block, n_swap are
    declared static so the loop bounds are concrete.

    Arguments
    ---------
    key                          : PRNGKey
    B, O                         : current state, see glossary
    params                       : (5,) parameter vector
    U, C, CH, WO, HT, HC         : participant data, see glossary
    n_block, n_swap, n_ffbs      : ints, iteration counts per stage

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

    # 3) n_block block-move MH iterations on B (legacy, default 0).
    def block_step(i, carry):
        key, B = carry
        key, sub = random.split(key)
        B = block_move(sub, B, O, U, C, CH, WO, params, HT, HC)
        return key, B
    key, B = lax.fori_loop(0, n_block, block_step, (key, B))

    # 4) n_swap swap-move MH iterations on B.
    def swap_step(i, carry):
        key, B = carry
        key, sub = random.split(key)
        B = swap_move(sub, B, O, U, C, CH, WO, params)
        return key, B
    key, B = lax.fori_loop(0, n_swap, swap_step, (key, B))
    return key, B, O


# ---------- per-trial sufficient stats + loglik ----------

@jax.jit
def stats_arrays(B, O, U, C, CH, WO):
    """Compute the per-trial sufficient statistics consumed by loglik().

    Splits the work into emission stats (per trial) and transition stats
    (per pair of consecutive trials), plus a few scalar counts over the
    order trajectory. With these in hand, loglik() can be evaluated for
    many (lam, eps_o, kappa, kappa_s, p_commit_o) combinations without
    re-running survivor / movetype logic.

    Arguments
    ---------
    B, O, U, C, CH, WO : see glossary at the top of the file

    Returns
    -------
    nS          : float (T,)   survivor count per trial (or 4 if no info)
    surv        : float (T,)   1 if chosen survived (or 1 if no info)
    mt          : int   (T,)   move type per trial (mt[0] = 0 by convention)
    nsa         : float (T,)   n available single-word moves per trial (nsa[0] = 0)
    nwa         : float (T,)   n available two-word swaps per trial (nwa[0] = 0)
    n_unk_stay  : int   ()     # (t-1, t) pairs both UNK
    n_commit    : int   ()     # pairs UNK -> committed
    n_o_stay    : int   ()     # pairs committed -> same committed order
    n_o_switch  : int   ()     # pairs committed -> different committed order
    """
    
    T = B.shape[0]
    # per-trial emission stats
    nS, surv = jax.vmap(emit_stat, in_axes=(0, 0, 0, 0, 0, None))(
        O, B, U, C, CH, WO)

    # per-pair (t-1, t) transition stats:
    def per_pair(t):
        mt = _movetype(B[t - 1], B[t], U[t])
        ns, nw = _count_moves(B[t - 1], U[t])
        return mt, ns.astype(jnp.float32), nw.astype(jnp.float32)
    
    # NOTE: per_pair internally uses B and U, 
    # even though they are not passed as arguments!
    mt, nsa, nwa = jax.vmap(per_pair)(jnp.arange(1, T))

    # Pad with a 0 at t=0 so that mt/nsa/nwa have length T 
    # (matching nS/surv).
    mt = jnp.concatenate([jnp.zeros(1, dtype=mt.dtype), mt])
    nsa = jnp.concatenate([jnp.zeros(1, dtype=nsa.dtype), nsa])
    nwa = jnp.concatenate([jnp.zeros(1, dtype=nwa.dtype), nwa])
    
    # order transition counts:
    op = O[:-1]
    oc = O[1:]
    # stay in UNK
    n_unk_stay = jnp.sum((op == UNK_O) & (oc == UNK_O))
    # commit from UNK to a committed order
    n_commit = jnp.sum((op == UNK_O) & (oc != UNK_O))
    # stay in the same committed order
    n_o_stay = jnp.sum((op != UNK_O) & (oc == op))
    # switch between two different committed orders
    n_o_switch = jnp.sum((op != UNK_O) & (oc != UNK_O) & (oc != op))
    
    return nS, surv, mt, nsa, nwa, n_unk_stay, n_commit, n_o_stay, n_o_switch


@jax.jit
def loglik(params, nS, surv, mt, nsa, nwa, n_unk_stay, n_commit, n_o_stay, n_o_switch):
    """Data log-likelihood given sufficient stats and parameters.

    Three terms summed:
        em : emission likelihood under lapse mixture (per trial)
        tr : belief-transition likelihood (per trial)
        ot : order-transition likelihood (over the trajectory)

    Arguments
    ---------
    params : float (5,)             (lam, eps_o, kappa, kappa_s, p_commit_o)
    nS, surv, mt, nsa, nwa          per-trial stats from stats_arrays
    n_unk_stay, n_commit,           scalar transition counts from stats_arrays
      n_o_stay, n_o_switch

    Returns
    -------
    ll : float ()                   total data log-likelihood
    """

    lam, eps_o, kappa, kappa_s, p_commit_o = params

    # emission probability:
    # Per-trial choice prob under the lapse mixture, then sum logs
    # surv is 1 if the chosen scene survived, 0 otherwise
    # nS is the number of surviving candidate scenes per trial
    p_per_trial = (1.0 - lam) * surv / nS + lam * 0.25
    em = jnp.log(p_per_trial).sum()

    # belief-transition probability:
    # Z is the per-trial normalizer of the softmax over the move types,
    # where the choice happens at each trial.
    # nsa is the number of feasible single-word moves per trial,
    # nwa similar for two-word swaps.
    # (calculated in _count_moves)
    # 1 is exp(0), i.e., the log-probability of no move, 
    # which is always there.
    # (= 1 + nsa * e^-kappa + nwa * e^-kappa_s).
    # NOTE: a consequence of this is that the probability of no move 
    # goes down when there are more feasible moves (since Z increases).
    Z = 1.0 + nsa * jnp.exp(-kappa) + nwa * jnp.exp(-kappa_s)
    cost = jnp.where(mt == 0, 0.0,
            # cost of a single-word move
            jnp.where(mt == 1, kappa,
              # cost of a two-word swap
              jnp.where(mt == 2, kappa_s, 
                # cost of a move that is not a single-word move 
                # or a two-word swap (which is not allowed, so inf)
                jnp.inf)))
    # mt is the move type per trial
    # mt[0] = 0 by construction, 
    # so its (cost=0, log Z=0) contribution is 0.
    # log P = log( exp(-cost) / Z ) = -cost - log Z
    tr = (-cost - jnp.log(Z)).sum()

    # order-transition probability:
    # Each pair (O[t-1], O[t]) contributes log of its M-entry
    # (which is the probability of the transition).
    # we just count the four equivalence classes and multiply.
    # NOTE: the numbers are calculated in stats_arrays based on O.
    # xlogy(n, p) = n * log(p) with xlogy(0, 0) = 0: when a sigmoid
    # saturates in float32 (p == 0 or 1 exactly) the log is -inf, and a plain
    # product with a zero count would give nan (0 * -inf) and make the MH
    # step reject unconditionally instead of using the finite ratio.
    ot = (xlogy(n_unk_stay, 1.0 - p_commit_o)
          + xlogy(n_commit, p_commit_o / N_O)
          + xlogy(n_o_stay, 1.0 - eps_o)
          + xlogy(n_o_switch, eps_o / (N_O - 1)))
    
    return em + tr + ot
