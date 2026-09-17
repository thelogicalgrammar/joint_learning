"""
Step 1 — hierarchical Gibbs fit of the partial-lexicon HMM on all participants.

All the machinery (batched data, kernels, run_chain, save_fit) lives in
jointlearn/hmm/sampler.py and is configured through the env vars documented
there (N_OUTER, BURN, THIN, N_CHAINS, N_FFBS, N_SWAP, N_BLOCK, CKPT_EVERY, RESUME;
EA_RESULTS_DIR for the output folder). Writes results/hierarchical_fit.pkl.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from jointlearn.hmm import sampler as S, model as NB, io as IO

if __name__ == '__main__':
    out_dir = IO.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Hierarchical fit (condition-level hyperprior), JAX vmap on '
          f'{jax.devices()[0]}.')
    print(f'  {S.N_CHAINS} chains x {S.N_OUTER} outer iters x {S.P_all} participants '
          f'x {S.N_COND} conditions (S.N_TPAR={S.N_TPAR}, kappa_s fixed at '
          f'{NB.KAPPA_S_FIXED})', flush=True)

    print('  compiling S.sweep_all + S.stats_all + S.mh_all (warm-up) ...',
          flush=True)
    t_c = time.time()
    # (named *_warm, not B0/O0: `B0` is the InvGamma scale imported from
    # config and used as a default argument of update_hyper.)
    key_warm = random.PRNGKey(0)
    B_warm = jnp.full((S.P_all, S.T, 7), -1, dtype=jnp.int8)
    O_warm = jnp.full((S.P_all, S.T), S.UNK_O, dtype=jnp.int8)
    tpar_warm = jnp.asarray(np.tile(S.HYPER_INIT[:, 0], (S.P_all, 1)),
                            dtype=jnp.float32)
    hyp_warm = S.init_hyper()
    k1, B1, O1 = S.sweep_all(key_warm, B_warm, O_warm, tpar_warm)
    s1 = S.stats_all(B1, O1)
    _ = S.mh_all(k1, tpar_warm, s1,
               jnp.asarray(hyp_warm['mu'][S.COND], dtype=jnp.float32),
               jnp.asarray(hyp_warm['sigma'], dtype=jnp.float32))
    print(f'    compiled in {time.time() - t_c:.1f}s', flush=True)

    t0 = time.time()
    chains = []
    for k in range(S.N_CHAINS):
        ck_path = str(out_dir / f'hier_chain{k}_ckpt.pkl')
        ch = S.run_chain(chain_id=k, seed=10000 + 100 * k, ckpt_path=ck_path)
        chains.append(ch)
    print(f'\nAll chains done in {(time.time() - t0) / 60:.1f} min', flush=True)

    out = out_dir / IO.FIT_NAME
    S.save_fit(chains, out)
    print(f'Saved: {out}')
    S.print_condition_means(chains)
    print('done')
