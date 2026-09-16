"""
Shared helpers used by the scripts in this directory.

Paths / IO
    RESULTS_DIR, FIGURES_DIR   — default `results/`, `figures/` next to this
                                 file; override with env vars EA_RESULTS_DIR,
                                 EA_FIGURES_DIR (used by tests / smoke runs
                                 so nothing in the project folders is touched)
    load_pickle, save_pickle   — pickle IO relative to RESULTS_DIR
    load_fit                   — load hierarchical_fit.pkl and check its format
    fit_signature, write_fit_signature, load_derived
                               — staleness guard: every pickle derived from a
                                 fit carries the fit's signature, checked
                                 against the sidecar `hierarchical_fit.sig.json`
    pool_chains, pool_hyper    — concatenate per-chain samples across chains
    savefig                    — save a figure into FIGURES_DIR

Analysis
    bag_consistent_mask        — candidates consistent with the utterance up to word order
    compute_swap_mask          — the word-order-discriminating trials
    pack_choices, unpack_choices — 2-bit packing of 4AFC choice sequences
    settle_matrix              — first-passage times of sampled O to the true order
    km_frame                   — per-participant survival frame for lifelines

Plotting
    violins                    — coloured violins with median + CI bars
"""
import os
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
import data

HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get('EA_RESULTS_DIR', HERE / 'results'))
FIGURES_DIR = Path(os.environ.get('EA_FIGURES_DIR', HERE / 'figures'))
FIT_NAME = 'hierarchical_fit.pkl'
SIG_NAME = 'hierarchical_fit.sig.json'
SIG_KEY = 'fit_signature'          # key under which derived pickles store it


# ---------------------------------------------------------------- IO ----

def load_pickle(name, directory=RESULTS_DIR):
    path = Path(directory) / name
    if not path.exists():
        raise SystemExit(f'Missing {path}. Run the upstream script first '
                         f'(see README.md, "Pipeline").')
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, name, directory=RESULTS_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved: {path}', flush=True)
    return path


def load_fit(name=FIT_NAME):
    """Load the hierarchical fit and refuse pickles that do not match the
    current code or data: older model versions (pooled hyperprior with
    array-valued `hyper`, or the `traj_store` format), a different number of
    fitted parameters, a different fixed kappa_s, or a different participant
    set / condition vector than `data` currently loads (every downstream
    script indexes the fit's samples by the current data)."""
    import config as CFG
    import model as NB
    fit = load_pickle(name)
    ch = fit['chains'][0]
    if 'B' not in ch or not isinstance(ch.get('hyper'), dict):
        raise SystemExit(
            f'{name} was produced by an older fit_hierarchical.py (no full '
            f'B/O samples and/or no condition-level hyper dict). Re-run '
            f'fit_hierarchical.py; old pickles are not compatible.')
    if fit.get('n_tpar') != CFG.N_TPAR:
        raise SystemExit(
            f'{name} has n_tpar={fit.get("n_tpar")} but config.N_TPAR='
            f'{CFG.N_TPAR}. Re-run fit_hierarchical.py.')
    if fit.get('kappa_s_fixed') != NB.KAPPA_S_FIXED:
        raise SystemExit(
            f'{name} was fitted with kappa_s_fixed={fit.get("kappa_s_fixed")} '
            f'but model.KAPPA_S_FIXED={NB.KAPPA_S_FIXED}. Re-run '
            f'fit_hierarchical.py.')
    cond = np.asarray(fit.get('cond', []))
    if cond.shape != data.true_order.shape or not np.array_equal(cond, data.true_order):
        raise SystemExit(
            f'{name} was fitted on {cond.shape[0]} participants whose condition '
            f'vector differs from the {data.true_order.shape[0]} participants '
            f'`data` currently loads (exclusion rule or CSV changed?). Re-run '
            f'fit_hierarchical.py on the current data.')
    return fit


def fit_signature(fit):
    """Small content-derived identity of a fit: run config, chain seeds and a
    hash of the retained tpar samples. Two fits with the same signature are
    the same fit for every downstream purpose."""
    chains = fit['chains']
    h = hashlib.sha1()
    for ch in chains:
        h.update(np.ascontiguousarray(ch['tpar'], dtype=np.float32).tobytes())
    return dict(n_outer=int(fit['n_outer']), burn=int(fit['burn']),
                thin=int(fit['thin']), n_chains=len(chains),
                seeds=[int(ch['seed']) for ch in chains],
                n_samples=int(sum(ch['tpar'].shape[0] for ch in chains)),
                tpar_sha1=h.hexdigest())


def write_fit_signature(fit, directory=RESULTS_DIR):
    """Write the sidecar next to the fit pickle (called by fit_hierarchical.
    save_fit and by infer_per_participant, so older fits get one too)."""
    sig = fit_signature(fit)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / SIG_NAME, 'w') as f:
        json.dump(sig, f, indent=1)
    return sig


def load_derived(name, directory=RESULTS_DIR):
    """Load a pickle written by infer_per_participant.py and refuse it if it
    was derived from a different fit than the one the sidecar describes
    (i.e. the fit was re-run but infer_per_participant.py was not)."""
    obj = load_pickle(name, directory)
    sig = obj.get(SIG_KEY) if isinstance(obj, dict) else None
    if sig is None:
        raise SystemExit(f'{name} carries no fit signature (written by an older '
                         f'infer_per_participant.py). Re-run infer_per_participant.py.')
    sidecar = Path(directory) / SIG_NAME
    if not sidecar.exists():
        print(f'[warn] {sidecar.name} missing: cannot check whether {name} is '
              f'stale; run infer_per_participant.py to (re)create it.', flush=True)
        return obj
    with open(sidecar) as f:
        current = json.load(f)
    if sig != current:
        raise SystemExit(
            f'{name} was derived from a different fit than {FIT_NAME} '
            f'(seeds {sig.get("seeds")} / {sig.get("n_samples")} samples vs '
            f'{current.get("seeds")} / {current.get("n_samples")}). '
            f'Re-run infer_per_participant.py.')
    return obj


def pool_chains(fit, key):
    """Concatenate per-chain arrays `chain[key]` along the sample axis."""
    return np.concatenate([ch[key] for ch in fit['chains']], axis=0)


def pool_hyper(fit, key):
    """Concatenate per-chain hyper arrays `chain['hyper'][key]` (mu, sigma, m, tau)."""
    return np.concatenate([ch['hyper'][key] for ch in fit['chains']], axis=0)


def savefig(fig, name, dpi=130):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
    print(f'Saved: {path}', flush=True)
    return path


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
