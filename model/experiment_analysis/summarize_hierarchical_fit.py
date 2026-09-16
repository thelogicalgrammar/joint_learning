"""
Analyse the multi-chain hierarchical fit (condition-level hyperprior).

  (1) Per-participant R-hat on logit(lambda) across chains, and R-hat on
      the condition means mu[w, k] and population means m_k.
  (2) Per participant in TRAJ_SUBSET, do the independent (multi-seed)
      chains agree on the final committed order? (mode-stuckness check;
      chains differ only by seed — O inits are overwritten by FFBS.)
  (3) Summary figure: R-hat histogram, per-chain population-lambda trace,
      TRAJ_SUBSET final orders per chain.

Requires a fit pickle produced by the current fit_hierarchical.py (full
B/O arrays, dict-valued `hyper`).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
import data as RP
import helpers as HLP
import model as NB
import config as CFG

order_labels = RP.order_labels_full
order_colors = RP.order_colors_full

fit = HLP.load_fit()
chains = fit['chains']
PARTS = fit['parts']
acc = fit['acc']
n_chains = len(chains)
n_post = chains[0]['tpar'].shape[0]
P = len(PARTS)
param_names = fit['param_names']
cond_labels = fit['cond_labels']

print(f'{n_chains} chains, {n_post} post-burn samples each, {P} participants')
print('MH acceptance rate (tpar) per chain [per coordinate '
      + '/'.join(fit['param_names']) + ']: '
      + ', '.join('/'.join(f'{r:.2f}' for r in np.atleast_1d(ch.get('mh_accept_rate', np.nan)))
                  for ch in chains))
print('group-shift acceptance rate per chain [per coordinate]: '
      + ', '.join('/'.join(f'{r:.2f}' for r in np.atleast_1d(ch.get('shift_accept_rate', np.nan)))
                  for ch in chains))
print('population-shift acceptance rate per chain [per coordinate]: '
      + ', '.join('/'.join(f'{r:.2f}' for r in np.atleast_1d(ch.get('pop_shift_accept_rate', np.nan)))
                  for ch in chains))


def chain_label(ch):
    return f'seed {ch.get("seed", ch["chain_id"])}'


def _ds(arr):
    dims = ('chain', 'draw') + tuple(f'd{i}' for i in range(arr.ndim - 2))
    return xr.Dataset({'x': (dims, arr)})


def rhat_nd(arr):
    """arr (C, S, ...) -> R-hat over the trailing dims."""
    return az.rhat(_ds(arr))['x'].values


def ess_nd(arr):
    """arr (C, S, ...) -> bulk effective sample size (all chains pooled)."""
    return az.ess(_ds(arr))['x'].values


def table(title, rhat, ess):
    """rhat/ess (W, K) tables per condition + population row given separately."""
    print(f'\n{title}')
    print(f'  {"cond":>5}  ' + '  '.join(f'{n:>22}' for n in param_names))
    for w, lbl in enumerate(cond_labels):
        print(f'  {lbl:>5}  ' + '  '.join(f'{rhat[w, k]:>13.3f} / {ess[w, k]:>6.0f}'
                                        for k in range(len(param_names))))


# --- per-participant logit(lambda): R-hat and ESS ---
tpar_all = np.stack([ch['tpar'] for ch in chains])              # (C, S, P, K)
n_draws_total = n_chains * n_post
lam_rhat = rhat_nd(tpar_all[..., 0])                            # (P,)
lam_ess = ess_nd(tpar_all[..., 0])                              # (P,)
print(f'\nPer-participant logit(lambda), {n_draws_total} pooled draws:')
print(f'  R-hat: median {np.nanmedian(lam_rhat):.3f}, '
      f'90th pctl {np.nanpercentile(lam_rhat, 90):.3f}, max {np.nanmax(lam_rhat):.3f}; '
      f'> 1.1: {(lam_rhat > 1.1).sum()}/{P}, > 1.2: {(lam_rhat > 1.2).sum()}/{P}')
print(f'  ESS:   median {np.nanmedian(lam_ess):.0f}, '
      f'10th pctl {np.nanpercentile(lam_ess, 10):.0f}, min {np.nanmin(lam_ess):.0f}; '
      f'< 100: {(lam_ess < 100).sum()}/{P}')

# --- condition means and population means: R-hat and ESS ---
if n_chains >= 2:
    mu_all = np.stack([ch['hyper']['mu'] for ch in chains])     # (C, S, W, K)
    m_all = np.stack([ch['hyper']['m'] for ch in chains])       # (C, S, K)
    tau_all = np.stack([ch['hyper']['tau'] for ch in chains])   # (C, S, K)
    table('Condition means mu[w, k]:  R-hat / ESS', rhat_nd(mu_all), ess_nd(mu_all))
    for name, arr in [('population m', m_all), ('tau', tau_all)]:
        r, e = rhat_nd(arr), ess_nd(arr)
        print(f'  {name:>13}  ' + '  '.join(f'{r[k]:>13.3f} / {e[k]:>6.0f}'
                                          for k in range(len(param_names))))
else:
    print('\n(single chain: no R-hat on hyperparameters)')

# --- per-participant final order per chain (TRAJ_SUBSET) ---
print('\n=== TRAJ_SUBSET: modal final order per chain ===')
print(f'{"pid":>5} {"acc":>5} {"true":>5} ', end='')
for c in range(n_chains):
    print(f' ch{c}({chain_label(chains[c])})', end='')
print()

true_orders = RP.true_order
TRAJ_SUBSET = fit['traj_subset']
parts_list = PARTS.tolist()

agreement = {}
for p in TRAJ_SUBSET:
    if p not in parts_list:
        continue
    p_idx = parts_list.index(p)
    row = [f'{p:>5} {acc[p_idx]:>4.0%} {order_labels[true_orders[p]]:>5}']
    final_orders = []
    for c in range(n_chains):
        # last retained sample for this chain + ppt: O of shape (T,)
        last_O = chains[c]['O'][-1, p_idx]
        # modal of last 50 trials, restricted to committed (0..5)
        tail = last_O[-50:]
        comm = tail[tail < 6]
        if len(comm) == 0:
            final = 6  # still UNK
        else:
            final = int(np.bincount(comm, minlength=6).argmax())
        final_orders.append(final)
        marker = '*' if final == true_orders[p] else ' '
        row.append(f'  {order_labels[final]:>5}{marker}')
    agreement[p] = final_orders
    print(''.join(row))

# --- figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) R-hat distribution
ax = axes[0]
ax.hist(lam_rhat[np.isfinite(lam_rhat)], bins=40, color='#377eb8', alpha=0.85)
ax.axvline(1.1, color='r', ls='--', lw=1, label='1.1')
ax.axvline(1.2, color='darkred', ls='--', lw=1, label='1.2')
ax.set_xlabel(r'$\hat{R}$ on logit($\lambda$) per participant')
ax.set_ylabel('# participants')
ax.set_title('Mixing: R-hat distribution')
ax.legend(fontsize=8)

# (b) per-chain population lambda trace
ax = axes[1]
for c, ch in enumerate(chains):
    k_lam = CFG.PARAM_NAMES.index('logit_lam')
    lam_pop = NB.tpar_to_nat_np(ch['hyper']['m'][:, k_lam], k_lam)   # -> natural
    ax.plot(lam_pop, alpha=0.7, lw=1.0, label=chain_label(ch))
ax.set_xlabel('post-burn sample idx')
ax.set_ylabel(r'population $\lambda$ (m)')
ax.set_title('Hyperposterior trace per chain')
ax.legend(fontsize=7, ncol=2)

# (c) per-chain TRAJ_SUBSET final orders heatmap
ax = axes[2]
pids = list(agreement.keys())
if pids:
    mat = np.array([agreement[p] for p in pids])      # (n_pids, n_chains)
    ax.imshow(mat, aspect='auto',
              cmap=matplotlib.colors.ListedColormap(order_colors),
              vmin=0, vmax=6)
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f'p{p}\n(true {order_labels[true_orders[p]]})'
                        for p in pids], fontsize=8)
    ax.set_xticks(range(n_chains))
    ax.set_xticklabels([chain_label(chains[c]) for c in range(n_chains)],
                       fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, order_labels[v], ha='center', va='center',
                    fontsize=7, color='white' if v < 6 else 'black',
                    fontweight='bold' if v == true_orders[pids[i]] else 'normal')
ax.set_xlabel('chain')
ax.set_title('Final committed order per chain')

plt.tight_layout()
HLP.savefig(fig, 'hierarchical_fit_summary.png')
