"""
Step 1 — hierarchical HMC-within-Gibbs fit of the partial-lexicon HMM on all
participants.

The sampler lives in jointlearn/hmm/sampler.py and is configured through env
vars documented there (N_WARMUP, N_SAMPLES, THIN, N_CHAINS, TRAJ, SEED0,
RESUME; EA_RESULTS_DIR for the output folder). Defaults = the reported fit.

Each chain runs in its own process (several chains in one process exhaust a
4 GB GPU): this script calls itself with CHAIN=k for k = 0..N_CHAINS-1, each
writing results/hier_chain{k}.pkl, then assembles them into
results/hierarchical_fit.pkl. With RESUME=1 finished chain files with the
same config are reused. `CHAIN=k python 01_fit.py` runs one chain only
(e.g. to spread chains over several GPUs or jobs; run 01_fit.py with
RESUME=1 afterwards to assemble).
"""
import os
import sys
import time
import subprocess
from pathlib import Path

from jointlearn.hmm import sampler as S, model as NB, io as IO

if __name__ == '__main__':
    out_dir = IO.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if 'CHAIN' in os.environ:                       # worker: one chain
        import jax
        k = int(os.environ['CHAIN'])
        print(f'[chain {k}] on {jax.devices()[0]}', flush=True)
        p = S.save_chain(S.run_chain(k), out_dir)
        print(f'[chain {k}] saved {p}', flush=True)
        sys.exit(0)

    print(f'Hierarchical fit (condition-level hyperprior), {S.SAMPLER}.')
    print(f'  {S.N_CHAINS} chains x ({S.N_WARMUP} warm-up + {S.N_SAMPLES} iterations, '
          f'thin {S.THIN}), trajectory length {S.TRAJ} x U[0.5, 1.5], {S.P_all} participants '
          f'x {S.N_COND} conditions (N_TPAR={S.N_TPAR}, kappa_s fixed at {NB.KAPPA_S_FIXED})',
          flush=True)
    t0 = time.time()
    chains = []
    for k in range(S.N_CHAINS):
        ch = S.load_chain(out_dir, k) if S.RESUME else None
        if ch is not None:
            print(f'[chain {k}] reusing {S.chain_path(out_dir, k).name}', flush=True)
        else:
            r = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve())],
                               env=dict(os.environ, CHAIN=str(k)))
            if r.returncode:
                raise SystemExit(f'chain {k} failed (exit {r.returncode})')
            ch = S.load_chain(out_dir, k)
            assert ch is not None, f'chain {k} did not write a matching chain file'
        chains.append(ch)
    print(f'\nAll chains done in {(time.time() - t0) / 60:.1f} min', flush=True)

    out = out_dir / IO.FIT_NAME
    S.save_fit(chains, out)
    print(f'Saved: {out}')
    print('done')
