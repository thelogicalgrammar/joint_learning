"""
IO plumbing shared by the HMM pipeline scripts (analyses/hmm).

    RESULTS_DIR, FIGURES_DIR   — default analyses/hmm/results, analyses/hmm/figures;
                                 override with env vars EA_RESULTS_DIR, EA_FIGURES_DIR
                                 (smoke runs, parameter recovery)
    load_pickle, save_pickle   — pickle IO relative to RESULTS_DIR
    load_fit                   — load hierarchical_fit.pkl and check its format
    fit_signature, write_fit_signature, load_derived
                               — staleness guard: every pickle derived from a
                                 fit carries the fit's signature, checked
                                 against the sidecar `hierarchical_fit.sig.json`
    pool_chains, pool_hyper    — concatenate per-chain samples across chains
    savefig                    — save a figure into FIGURES_DIR
"""
import os
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
from jointlearn.hmm import dataset as data

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1] / 'analyses' / 'hmm'
RESULTS_DIR = Path(os.environ.get('EA_RESULTS_DIR', ANALYSIS_DIR / 'results'))
FIGURES_DIR = Path(os.environ.get('EA_FIGURES_DIR', ANALYSIS_DIR / 'figures'))
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
    from jointlearn.hmm import config as CFG, model as NB
    fit = load_pickle(name)
    ch = fit['chains'][0]
    if 'B' not in ch or not isinstance(ch.get('hyper'), dict):
        raise SystemExit(
            f'{name} was produced by an older 01_fit.py (no full '
            f'B/O samples and/or no condition-level hyper dict). Re-run '
            f'01_fit.py; old pickles are not compatible.')
    if fit.get('n_tpar') != CFG.N_TPAR:
        raise SystemExit(
            f'{name} has n_tpar={fit.get("n_tpar")} but config.N_TPAR='
            f'{CFG.N_TPAR}. Re-run 01_fit.py.')
    if fit.get('kappa_s_fixed') != NB.KAPPA_S_FIXED:
        raise SystemExit(
            f'{name} was fitted with kappa_s_fixed={fit.get("kappa_s_fixed")} '
            f'but model.KAPPA_S_FIXED={NB.KAPPA_S_FIXED}. Re-run '
            f'01_fit.py.')
    cond = np.asarray(fit.get('cond', []))
    if cond.shape != data.true_order.shape or not np.array_equal(cond, data.true_order):
        raise SystemExit(
            f'{name} was fitted on {cond.shape[0]} participants whose condition '
            f'vector differs from the {data.true_order.shape[0]} participants '
            f'`data` currently loads (exclusion rule or CSV changed?). Re-run '
            f'01_fit.py on the current data.')
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
    """Write the sidecar next to the fit pickle (called by sampler.
    save_fit and by 02_postprocess, so older fits get one too)."""
    sig = fit_signature(fit)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / SIG_NAME, 'w') as f:
        json.dump(sig, f, indent=1)
    return sig


def load_derived(name, directory=RESULTS_DIR):
    """Load a pickle written by 02_postprocess.py and refuse it if it
    was derived from a different fit than the one the sidecar describes
    (i.e. the fit was re-run but 02_postprocess.py was not)."""
    obj = load_pickle(name, directory)
    sig = obj.get(SIG_KEY) if isinstance(obj, dict) else None
    if sig is None:
        raise SystemExit(f'{name} carries no fit signature (written by an older '
                         f'02_postprocess.py). Re-run 02_postprocess.py.')
    sidecar = Path(directory) / SIG_NAME
    if not sidecar.exists():
        print(f'[warn] {sidecar.name} missing: cannot check whether {name} is '
              f'stale; run 02_postprocess.py to (re)create it.', flush=True)
        return obj
    with open(sidecar) as f:
        current = json.load(f)
    if sig != current:
        raise SystemExit(
            f'{name} was derived from a different fit than {FIT_NAME} '
            f'(seeds {sig.get("seeds")} / {sig.get("n_samples")} samples vs '
            f'{current.get("seeds")} / {current.get("n_samples")}). '
            f'Re-run 02_postprocess.py.')
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
