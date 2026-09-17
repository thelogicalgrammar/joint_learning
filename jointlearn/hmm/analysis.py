"""
Analysis helpers shared by the HMM pipeline scripts (analyses/hmm).

    bag_consistent_mask        — candidates consistent with the utterance up to word order
    compute_swap_mask          — the word-order-discriminating trials (~9%)
    pack_choices, unpack_choices — 2-bit packing of 4AFC choice sequences
    settle_matrix              — first-passage times of sampled O to the true order
    km_frame                   — per-participant survival frame for lifelines
    violins                    — coloured violins with median + CI bars
"""
import numpy as np
from jointlearn.hmm import dataset as data


# ---------------------------------------------------------- analysis ----

def bag_consistent_mask():
    """Per (trial, participant, candidate): True iff the candidate scene's
    meaning bag contains the bag-of-words decoding of the signal under the
    participant's TRUE lexicon, i.e. the candidate is consistent with the
    utterance up to word order. Candidate 0 (the correct scene) always is."""
    sig, cand = data.signals, data.cand_mean        # (T, P, 3), (T, P, 4, 3)
    P = sig.shape[1]
    # decoded meanings of the 3 uttered words under the true lexicon
    dec = data.inv_true[np.arange(P)[None, :, None], sig]        # (T, P, 3)
    dec_set = np.zeros(sig.shape[:2] + (7,), dtype=bool)         # (T, P, 7)
    np.put_along_axis(dec_set, dec, True, axis=-1)
    cand_set = np.zeros(cand.shape[:3] + (7,), dtype=bool)       # (T, P, 4, 7)
    np.put_along_axis(cand_set, cand, True, axis=-1)
    # candidate j survives iff decoded set ⊆ candidate's meaning set
    return ~(dec_set[:, :, None, :] & ~cand_set).any(axis=-1)    # (T, P, 4)


def compute_swap_mask():
    """Per (trial, participant): True iff this is a role-swap trial,
    i.e. ≥ 2 of the 4 candidate scenes have a meaning bag that contains
    the bag-of-words decoded set under the participant's true lexicon.
    These are the ~9% of trials where word-order knowledge is needed."""
    return bag_consistent_mask().sum(axis=-1) >= 2


def pack_choices(ch):
    """4AFC choices (..., T) in 0..3 -> 2-bit packed uint8 (..., ceil(T/4)).
    Used to store posterior-predictive replicate choice sequences compactly
    (a (S, P, T) array packs to a quarter of a byte per trial)."""
    ch = np.asarray(ch)
    T = ch.shape[-1]
    pad = (-T) % 4
    if pad:
        ch = np.concatenate([ch, np.zeros(ch.shape[:-1] + (pad,), ch.dtype)], axis=-1)
    q = ch.reshape(ch.shape[:-1] + (-1, 4)).astype(np.uint8)
    return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).astype(np.uint8)


def unpack_choices(packed, T):
    """Inverse of pack_choices: (..., ceil(T/4)) uint8 -> (..., T) uint8 in 0..3."""
    packed = np.asarray(packed, dtype=np.uint8)
    q = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], axis=-1)
    return q.reshape(packed.shape[:-1] + (-1,))[..., :T]


def settle_matrix(o_samples, true_order):
    """First-passage times of sampled order trajectories to the true order.

    o_samples : int array (S, P, T) of posterior O samples, or the legacy
                dict {p: int array (S, T)}
    true_order: int array (P,) of each participant's true order

    Returns
    -------
    settle  : int16 (S, P)   first trial where O == true order (T if never)
    reached : bool  (S, P)   whether the trajectory ever reaches it
              (identically settle < T; returned for convenience)
    """
    if isinstance(o_samples, dict):
        o_samples = np.stack([o_samples[p] for p in range(len(o_samples))], axis=1)
    S, P, T = o_samples.shape
    mask = o_samples == np.asarray(true_order)[None, :, None]         # (S, P, T)
    reached = mask.any(axis=2)
    settle = np.where(reached, np.argmax(mask, axis=2), T).astype(np.int16)
    return settle, reached


def km_frame(settle_sp, true_order, T):
    """Per-participant survival frame from posterior settle times (S, P):
    event time = rounded posterior median, censored iff it equals T."""
    import pandas as pd
    settle = np.round(np.median(settle_sp, axis=0)).astype(int)
    return pd.DataFrame({'T': settle,
                         'event': (settle < T).astype(int),
                         'group': [data.order_labels[k] for k in true_order]})


# ---------------------------------------------------------- plotting ----

def violins(ax, samples, labels, colors, ci=(2.5, 97.5), width=0.8):
    """One violin per entry of `samples` (list of 1-D arrays) with a black
    median bar and a CI whisker."""
    # positions with no finite values are left empty (violinplot raises on them)
    keep = [i for i, x in enumerate(samples) if np.isfinite(x).sum() > 0]
    if keep:
        vio = ax.violinplot([np.asarray(samples[i])[np.isfinite(samples[i])] for i in keep],
                            positions=keep, widths=width, showextrema=False)
        for i, body in zip(keep, vio['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.55)
            body.set_edgecolor('black')
    for i in keep:
        x = np.asarray(samples[i])[np.isfinite(samples[i])]
        lo, hi = np.percentile(x, ci)
        ax.plot([i, i], [lo, hi], color='black', lw=1.0)
        ax.plot([i - 0.18, i + 0.18], [np.median(x)] * 2, color='black', lw=1.5)
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(labels)
