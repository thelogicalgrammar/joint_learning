"""
PPC diagnostics: residual-by-trial, lambda decomposition, swap-trial residual
by true order, and late-session residual by commitment status. Reads
per_participant_ppc.pkl + per_participant_marginals.pkl + raw choices,
writes 4 figures to figures/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import data as RP
import helpers as HLP

order_labels = RP.order_labels
typology = {w: g for g, idx in RP.TYPOLOGY.items() for w in idx}

ppc = HLP.load_derived('per_participant_ppc.pkl')
marg = HLP.load_derived('per_participant_marginals.pkl')

pred = ppc['pred_p_wrong']                          # (P, T)
swap_mask = ppc['swap_mask']                        # (P, T)
P, T = pred.shape
obs = (RP.choices.T != 0).astype(np.float32)        # (P, T)
true_order = RP.true_order
lam = np.array([marg[p]['params'][0] for p in range(P)])
# minimum dP(wrong)/dlam for a participant's mean wrong rate to say anything
# about lam (see fig_lambda_decomp); 0.2 means a 0.1 change in lam moves the
# predicted wrong rate by at least 0.02 (~4 trials out of 200).
LEVERAGE_MIN = 0.2


# ------------------------------------------------------------------
# Figure 1: residual by trial
# ------------------------------------------------------------------
def fig_residual_by_trial():
    obs_rate = obs.mean(axis=0)
    pred_rate = pred.mean(axis=0)
    resid = obs_rate - pred_rate
    slope, intercept, r, pval, se = stats.linregress(np.arange(T), resid)
    print(f'[fig1] residual ~ trial: slope = {slope * 1000:.3f} per 1k, p = {pval:.2e}')
    print(f'       early {resid[:50].mean():+.4f}, '
          f'mid {resid[50:150].mean():+.4f}, '
          f'late {resid[150:].mean():+.4f}')

    swap_obs_rate = np.where(swap_mask.sum(axis=0) > 0,
                             (obs * swap_mask).sum(axis=0) / swap_mask.sum(axis=0),
                             np.nan)
    swap_pred_rate = np.where(swap_mask.sum(axis=0) > 0,
                              (pred * swap_mask).sum(axis=0) / swap_mask.sum(axis=0),
                              np.nan)
    nonswap_obs_rate = (obs * ~swap_mask).sum(axis=0) / (~swap_mask).sum(axis=0)
    nonswap_pred_rate = (pred * ~swap_mask).sum(axis=0) / (~swap_mask).sum(axis=0)

    def smooth(x, w):
        """NaN-aware moving average: mean of the non-NaN values in each
        window (so NaN trials and the series ends are not pulled toward 0)."""
        ok = np.isfinite(x)
        num = np.convolve(np.where(ok, x, 0.0), np.ones(w), mode='same')
        den = np.convolve(ok.astype(float), np.ones(w), mode='same')
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax = axes[0]
    ax.plot(np.arange(T), smooth(obs_rate, 10), color='#d73027', lw=1.4, label='observed')
    ax.plot(np.arange(T), smooth(pred_rate, 10), color='#1a9850', lw=1.4, label='predicted')
    ax.set_ylabel('P(wrong) [10-trial moving avg]')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Per-trial error rate: observed vs PPC-predicted, averaged over participants')

    ax = axes[1]
    ax.plot(np.arange(T), smooth(resid, 10), color='#7f7f7f', lw=1.0, alpha=0.6,
            label=f'all (slope={slope * 1000:+.3f}/1k, p={pval:.1e})')
    ax.plot(np.arange(T), smooth(swap_obs_rate - swap_pred_rate, 10),
            color='#377eb8', lw=1.3, label='swap trials')
    ax.plot(np.arange(T), smooth(nonswap_obs_rate - nonswap_pred_rate, 10),
            color='#ff7f0e', lw=1.3, label='non-swap trials')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.plot(np.arange(T), intercept + slope * np.arange(T), color='#d73027',
            lw=1.0, ls=':', label='OLS fit (all)')
    ax.set_ylabel('obs - pred [10-trial MA]')
    ax.set_xlabel('trial')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    HLP.savefig(fig, 'ppc_residual_by_trial.png')
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: lambda decomposition
# ------------------------------------------------------------------
def fig_lambda_decomp():
    pred_mean = pred.mean(axis=1)
    obs_mean = obs.mean(axis=1)
    resid_ppt = obs_mean - pred_mean
    # Per participant, P(wrong) = (1 - lam) * q + 0.75 * lam, where q is the
    # mean belief-driven wrong rate. Back q out of the prediction, then solve
    # for the lapse rate that reproduces the observed wrong rate with q held
    # fixed. (The earlier (obs - (pred - 0.75 lam)) / 0.75 held (1 - lam) q
    # fixed instead and understated the needed lam whenever q is not ~0.)
    qbar = (pred_mean - 0.75 * lam) / (1.0 - lam)
    # dP(wrong)/dlam = 0.75 - qbar: for a participant whose beliefs already
    # predict near-chance performance (qbar ~ 0.75) the mean wrong rate
    # carries no information about lam and the solve is ill-conditioned, so
    # those participants are excluded from the needed-lam statistics.
    leverage = 0.75 - qbar
    identified = leverage > LEVERAGE_MIN
    needed_lam = np.where(identified, (obs_mean - qbar) / leverage, np.nan)
    slope, intercept, r, pval, _ = stats.linregress(lam, resid_ppt)

    print(f'\n[fig2] per-ppt: obs mean {obs_mean.mean():.3f}, '
          f'pred mean {pred_mean.mean():.3f}, resid mean {resid_ppt.mean():+.4f}')
    print(f'       resid ~ lam: slope = {slope:+.3f}, r = {r:+.3f}, p = {pval:.2e}')
    print(f'       lam identified from the mean wrong rate (0.75 - qbar > '
          f'{LEVERAGE_MIN}) for {identified.sum()}/{P} ppts; on those: '
          f'fitted lam mean = {lam[identified].mean():.3f}, '
          f'needed lam mean = {np.nanmean(needed_lam):.3f}, '
          f'median = {np.nanmedian(needed_lam):.3f}; '
          f'needed < 0: {(needed_lam < 0).sum()}/{identified.sum()}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.scatter(pred_mean, obs_mean, s=10, alpha=0.5, c='#377eb8')
    mx = max(pred_mean.max(), obs_mean.max())
    ax.plot([0, mx], [0, mx], 'k--', lw=0.7)
    ax.set_xlabel('predicted P(wrong), per ppt')
    ax.set_ylabel('observed P(wrong), per ppt')
    ax.set_title('Per-ppt mean wrong rate: obs vs pred')

    ax = axes[1]
    ax.scatter(lam, resid_ppt, s=10, alpha=0.5, c='#ff7f0e')
    xfit = np.linspace(lam.min(), lam.max(), 50)
    ax.plot(xfit, intercept + slope * xfit, 'k--', lw=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'fitted $\lambda$ per ppt')
    ax.set_ylabel('residual (obs - pred)')
    ax.set_title(f'Residual vs lambda  (r={r:+.2f}, p={pval:.1e})')

    ax = axes[2]
    ax.scatter(lam[identified], needed_lam[identified], s=10, alpha=0.5, c='#984ea3')
    mx2 = max(lam.max(), np.nanmax(needed_lam))
    mn2 = min(lam.min(), np.nanmin(needed_lam))
    ax.plot([mn2, mx2], [mn2, mx2], 'k--', lw=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'fitted $\lambda$')
    ax.set_ylabel(r'$\lambda$ needed to match obs')
    ax.set_title(f'Lambda: fitted vs needed-for-match '
                 f'({identified.sum()}/{P} ppts with identifiable $\\lambda$)')

    plt.tight_layout()
    HLP.savefig(fig, 'ppc_lambda_decomp.png')
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: swap-trial residual by true order
# ------------------------------------------------------------------
def fig_swap_by_order():
    resid_swap = np.full(P, np.nan, dtype=np.float32)
    n_swap = np.zeros(P, dtype=int)
    for p in range(P):
        sm = swap_mask[p]
        n_swap[p] = sm.sum()
        if n_swap[p] > 0:
            resid_swap[p] = obs[p, sm].mean() - pred[p, sm].mean()

    print('\n[fig3] swap-trial residual by order:')
    group_resid = {}
    for wo in range(6):
        mask = (true_order == wo) & ~np.isnan(resid_swap)
        r = resid_swap[mask]
        group_resid[wo] = r
        if len(r):
            ci = stats.t.interval(0.95, len(r) - 1, loc=r.mean(), scale=stats.sem(r))
            print(f'       {order_labels[wo]:>5} ({typology[wo]:>9}, n={len(r):>3}): '
                  f'{r.mean():+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]')

    s_init = np.isin(true_order, RP.TYPOLOGY['S-initial'])
    v_init = np.isin(true_order, RP.TYPOLOGY['V-initial'])
    o_init = np.isin(true_order, RP.TYPOLOGY['O-initial'])
    for name, m1, m2 in [('S-init vs rest', s_init, ~s_init),
                         ('SVO vs rest', true_order == 0, true_order != 0),
                         ('O-init vs rest', o_init, ~o_init)]:
        r1 = resid_swap[m1 & ~np.isnan(resid_swap)]
        r2 = resid_swap[m2 & ~np.isnan(resid_swap)]
        t, p_t = stats.ttest_ind(r1, r2, equal_var=False)
        u, p_u = stats.mannwhitneyu(r1, r2, alternative='two-sided')
        print(f'       {name}: diff = {r1.mean() - r2.mean():+.4f}, '
              f'Welch p = {p_t:.3f}, MW p = {p_u:.3f}')

    f, p = stats.f_oneway(*[group_resid[wo] for wo in range(6)])
    print(f'       6-way ANOVA: F = {f:.2f}, p = {p:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    HLP.violins(ax, [group_resid[wo] for wo in range(6)], order_labels,
                RP.order_colors, ci=(25, 75))
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('per-ppt swap-trial residual (obs - pred); bar = median, whisker = IQR')
    ax.set_title('Swap-trial residual by true order')

    ax = axes[1]
    for label, mask, c in [('S-init (SVO+SOV)', s_init, '#377eb8'),
                           ('V-init (VSO+VOS)', v_init, '#984ea3'),
                           ('O-init (OSV+OVS)', o_init, '#e41a1c')]:
        r = resid_swap[mask & ~np.isnan(resid_swap)]
        ax.hist(r, bins=20, alpha=0.55, color=c,
                label=f'{label} n={len(r)}, mean={r.mean():+.3f}')
    ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('per-ppt swap-trial residual')
    ax.set_ylabel('# ppts')
    ax.set_title('Distribution by typological group')
    ax.legend(fontsize=8)
    plt.tight_layout()
    HLP.savefig(fig, 'ppc_swap_by_order.png')
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 4: late-session residual by commitment status
# ------------------------------------------------------------------
def fig_late_by_commitment():
    WINDOW_START = 150
    late_om = np.array([marg[p]['order_marg'][WINDOW_START:].mean(axis=0)
                        for p in range(P)])
    modal_o = late_om.argmax(axis=1)
    cat = np.empty(P, dtype=object)
    for p in range(P):
        if modal_o[p] == 6:
            cat[p] = 'uncommitted'
        elif modal_o[p] == true_order[p]:
            cat[p] = 'committed_truth'
        else:
            cat[p] = 'committed_wrong'

    print(f'\n[fig4] commitment status at trials {WINDOW_START}+:')
    for k in ['committed_truth', 'committed_wrong', 'uncommitted']:
        print(f'       {k}: {(cat == k).sum()}/{P}')

    late_pred = pred[:, WINDOW_START:].mean(axis=1)
    late_obs = obs[:, WINDOW_START:].mean(axis=1)
    late_resid = late_obs - late_pred
    late_swap_resid = np.full(P, np.nan)
    late_nonswap_resid = np.full(P, np.nan)
    for p in range(P):
        sm = swap_mask[p, WINDOW_START:]
        pp = pred[p, WINDOW_START:]
        oo = obs[p, WINDOW_START:]
        if sm.any():
            late_swap_resid[p] = oo[sm].mean() - pp[sm].mean()
        if (~sm).any():
            late_nonswap_resid[p] = oo[~sm].mean() - pp[~sm].mean()

    print(f'       {"group":>20}  {"n":>4}  {"resid(all)":>11}  '
          f'{"resid(swap)":>12}  {"resid(nonsw)":>14}  '
          f'{"mean lam":>9}  {"obs late":>10}  {"pred late":>10}')
    for k in ['committed_truth', 'committed_wrong', 'uncommitted']:
        m = (cat == k)
        n = m.sum()
        if n == 0:
            continue
        print(f'       {k:>20}  {n:>4}  {late_resid[m].mean():>+11.4f}  '
              f'{np.nanmean(late_swap_resid[m]):>+12.4f}  '
              f'{np.nanmean(late_nonswap_resid[m]):>+14.4f}  '
              f'{lam[m].mean():>9.3f}  '
              f'{late_obs[m].mean():>10.3f}  {late_pred[m].mean():>10.3f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = ['committed_truth', 'committed_wrong', 'uncommitted']
    colors = ['#1a9850', '#d73027', '#7f7f7f']
    labels = {'committed_truth': 'committed→truth',
              'committed_wrong': 'committed→wrong',
              'uncommitted': 'still uncommitted'}
    for ax, vals, title in [(axes[0], late_resid, 'late residual (all trials)'),
                            (axes[1], late_swap_resid, 'late residual (swap)'),
                            (axes[2], late_nonswap_resid, 'late residual (non-swap)')]:
        for k, col in zip(groups, colors):
            m = (cat == k) & ~np.isnan(vals)
            if m.sum() == 0:
                continue
            ax.hist(vals[m], bins=20, alpha=0.55, color=col,
                    label=f'{labels[k]} n={m.sum()}, mean={vals[m].mean():+.3f}')
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('obs - pred')
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('# ppts')
    plt.tight_layout()
    HLP.savefig(fig, 'ppc_late_by_commitment.png')
    plt.close(fig)


# ------------------------------------------------------------------
# Top-line stats
# ------------------------------------------------------------------
def print_pvalues():
    p_t = ppc['p_total']
    p_s = ppc['p_swap']
    print(f'\n[ppc] median p_total = {np.median(p_t):.3f}, '
          f'median p_swap = {np.median(p_s):.3f}')
    for name, lo, hi in [('0.025/0.975', 0.025, 0.975),
                          ('0.05/0.95', 0.05, 0.95),
                          ('0.10/0.90', 0.10, 0.90)]:
        n_t = int(((p_t < lo) | (p_t > hi)).sum())
        n_s = int(((p_s < lo) | (p_s > hi)).sum())
        print(f'       at {name}: extreme p_total {n_t}/{P}, '
              f'extreme p_swap {n_s}/{P}')


if __name__ == '__main__':
    print_pvalues()
    fig_residual_by_trial()
    fig_lambda_decomp()
    fig_swap_by_order()
    fig_late_by_commitment()
    print('\ndone')
