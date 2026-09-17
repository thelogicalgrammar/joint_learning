"""
Step 4 — convergence and prior-vs-posterior diagnostics of the hierarchical fit.

convergence():           R-hat / ESS across chains on per-participant logit(lambda),
                         condition means and population means; MH acceptance per
                         chain; modal final order per chain for TRAJ_SUBSET; figure
                         hierarchical_fit_summary.png.
hyperprior_predictive(): new-participant prior vs posterior predictive per parameter
                         on the natural scale; figure hyperprior_predictive.png.

Reads results/hierarchical_fit.pkl.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm import model as NB
from jointlearn.hmm import config as CFG
from scipy.stats import gaussian_kde, invgamma


def convergence():
    order_labels = RP.order_labels_full
    order_colors = RP.order_colors_full

    fit = IO.load_fit()
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
    IO.savefig(fig, 'hierarchical_fit_summary.png')


def hyperprior_predictive():
    T = RP.T

    # Display settings per fitted coordinate, keyed by its config.PARAM_NAMES
    # entry (the transformed -> natural mapping comes from model.tpar_to_nat_np).
    #   (axis label, x range on the natural scale, log x axis)
    DISPLAY = {
        'logit_lam':        (r'$\lambda$  (lapse)', (0, 0.4), False),
        'log_kappa':        (r'$\kappa$  (single-move log-cost)', (0, 15), False),
        'logit_eps_o':      (r'$\varepsilon_o$  (order-switch rate per trial)', (1e-4, 0.1), True),
        'logit_p_commit_o': (r'$\gamma_o$  (commit rate per trial)', (1e-3, 0.5), True),
    }
    NAMES = [DISPLAY[n][0] for n in CFG.PARAM_NAMES]
    X_RANGES = [DISPLAY[n][1] for n in CFG.PARAM_NAMES]
    LOG_X = [DISPLAY[n][2] for n in CFG.PARAM_NAMES]


    # typical move counts for translating kappa to per-trial move probability
    N_SINGLE_TYP, N_SWAP_TYP = 15.0, 5.0


    def kappa_to_p_move(kappa, kappa_s=NB.KAPPA_S_FIXED):
        z = 1.0 + N_SINGLE_TYP * np.exp(-kappa) + N_SWAP_TYP * np.exp(-kappa_s)
        return 1.0 - 1.0 / z


    def translate(med, k):
        """Translate a natural-scale median into a practical sentence."""
        name = CFG.PARAM_NAMES[k]
        if name == 'logit_lam':
            return f'~{med * T:.1f} lapses / {T} trials'
        if name == 'log_kappa':
            return f'P(belief move / trial) ~ {kappa_to_p_move(med):.2%}'
        if name == 'logit_eps_o':
            return f'~{med * T:.2f} order switches / {T} trials'
        if name == 'logit_p_commit_o':
            if med <= 0:
                return ''
            t50 = np.log(0.5) / np.log1p(-med)
            return f'50% have committed by trial {t50:.0f}'


    # ---- load fit ----
    fit = IO.load_fit()
    n_tpar = fit['n_tpar']
    PRIOR_MU0, PRIOR_SD0 = fit['prior_mu0'], fit['prior_sd0']
    A0, B0, A_TAU, B_TAU = fit['a0'], fit['b0'], fit['a_tau'], fit['b_tau']

    mu_post = IO.pool_hyper(fit, 'mu')                     # (S, W, K)
    sig_post = IO.pool_hyper(fit, 'sigma')                 # (S, K)
    tpar_pmean = IO.pool_chains(fit, 'tpar').mean(axis=0)  # (P, K)
    S, W, _ = mu_post.shape

    rng = np.random.default_rng(0)
    N = 200000


    def sample_prior_predictive(k, n):
        """n new-participant draws on tpar dim k from the full prior hierarchy."""
        tau2 = invgamma.rvs(A_TAU, scale=B_TAU, size=n, random_state=rng)
        sig2 = invgamma.rvs(A0, scale=B0, size=n, random_state=rng)
        m = rng.normal(PRIOR_MU0[k], PRIOR_SD0[k], size=n)
        mu_w = rng.normal(m, np.sqrt(tau2))
        return rng.normal(mu_w, np.sqrt(sig2))


    def sample_posterior_predictive(k, n):
        """n new-participant draws on tpar dim k, integrating over the
        hyperposterior (a random sample s and a random condition w per draw)."""
        s = rng.integers(0, S, size=n)
        w = rng.integers(0, W, size=n)
        return rng.normal(mu_post[s, w, k], sig_post[s, k])


    fig, axes = plt.subplots(1, n_tpar, figsize=(3.4 * n_tpar, 4.6))
    for k in range(n_tpar):
        ax = axes[k]
        prior = NB.tpar_to_nat_np(sample_prior_predictive(k, N), k)
        post = NB.tpar_to_nat_np(sample_posterior_predictive(k, N), k)
        emp = NB.tpar_to_nat_np(tpar_pmean[:, k], k)

        lo, hi = X_RANGES[k]
        xx = np.geomspace(max(lo, 1e-8), hi, 400) if LOG_X[k] else np.linspace(lo, hi, 400)

        pr_in = prior[(prior >= lo) & (prior <= hi)]
        po_in = post[(post >= lo) & (post <= hi)]
        if len(pr_in) < 50 or len(po_in) < 50:
            ax.set_title(f'{NAMES[k]}\n(too few in range; check X_RANGES)')
            continue
        kd_pr = gaussian_kde(pr_in, bw_method=0.12)
        kd_po = gaussian_kde(po_in, bw_method=0.12)

        ymax = max(kd_pr(xx).max(), kd_po(xx).max())
        ax.fill_between(xx, kd_pr(xx), color='#3182bd', alpha=0.30, label='prior predictive')
        ax.plot(xx, kd_pr(xx), color='#3182bd', lw=1.4)
        ax.fill_between(xx, kd_po(xx), color='#e6550d', alpha=0.35, label='posterior predictive')
        ax.plot(xx, kd_po(xx), color='#e6550d', lw=1.4)

        m_pr, m_po = np.median(prior), np.median(post)
        ax.axvline(m_pr, color='#3182bd', ls='--', lw=1.0)
        ax.axvline(m_po, color='#e6550d', ls='--', lw=1.0)
        ax.scatter(emp, np.full(len(emp), -0.05 * ymax), marker='|', s=40,
                   color='black', alpha=0.35,
                   label=f'{len(emp)} ppts (post means)' if k == 0 else None)

        txt = (f'PRIOR  med = {m_pr:.4g}\n   {translate(m_pr, k)}\n\n'
               f'POST   med = {m_po:.4g}\n   {translate(m_po, k)}')
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
                ha='right', va='top', family='monospace',
                bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.85,
                          boxstyle='round,pad=0.3'))
        if LOG_X[k]:
            ax.set_xscale('log')
        ax.set_xlim(lo, hi)
        ax.set_xlabel(NAMES[k], fontsize=10)
        ax.set_ylabel('density' if k == 0 else '')
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=-0.10 * ymax)
        if k == 0:
            ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.5),
                      ncol=3, frameon=False)

    fig.suptitle('Hyperprior predictive: new-participant prior (blue) vs posterior '
                 '(orange) on natural scale, conditions pooled; black ticks are '
                 'per-ppt posterior means.', y=1.02, fontsize=11)
    plt.tight_layout()
    IO.savefig(fig, 'hyperprior_predictive.png')


if __name__ == '__main__':
    convergence()
    hyperprior_predictive()
