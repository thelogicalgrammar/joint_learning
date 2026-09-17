"""
Step 5 — posterior predictive checks (mixed PPC: replicates conditional on the
inferred latent trajectories; see 02_postprocess.py).

diagnostics(): residual by trial, lambda decomposition, swap-trial residual by
               condition, late-session residual by commitment status
               (ppc_residual_by_trial.png, ppc_lambda_decomp.png,
               ppc_swap_by_order.png, ppc_late_by_commitment.png).
pvalues():     posterior predictive p-values p = P(T(y_rep, theta) >= T(y_obs, theta))
               for the emission log-likelihood, block error counts, error
               clustering, swap-error structure and errors by survivor count
               (ppc_loglik.png, ppc_blocks.png, ppc_sequential.png,
               ppc_pvalue_calibration.png). The docstrings of the two functions
               describe the statistics.

Reads results/per_participant_ppc.pkl + per_participant_marginals.pkl.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN


def diagnostics():
    order_labels = RP.order_labels
    typology = {w: g for g, idx in RP.TYPOLOGY.items() for w in idx}

    ppc = IO.load_derived('per_participant_ppc.pkl')
    marg = IO.load_derived('per_participant_marginals.pkl')

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
        IO.savefig(fig, 'ppc_residual_by_trial.png')
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
        IO.savefig(fig, 'ppc_lambda_decomp.png')
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
        AN.violins(ax, [group_resid[wo] for wo in range(6)], order_labels,
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
        IO.savefig(fig, 'ppc_swap_by_order.png')
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
        IO.savefig(fig, 'ppc_late_by_commitment.png')
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
    print_pvalues()
    fig_residual_by_trial()
    fig_lambda_decomp()
    fig_swap_by_order()
    fig_late_by_commitment()
    print('\ndone')


def pvalues():
    BLOCK = 25
    ppc = IO.load_derived('per_participant_ppc.pkl')
    if 'rep_choice_packed' not in ppc:
        raise SystemExit('per_participant_ppc.pkl has no replicate choice sequences: '
                         're-run 02_postprocess.py')
    T = int(ppc['T'])
    rep = AN.unpack_choices(ppc['rep_choice_packed'], T)        # (S, P, T) uint8
    S, P, _ = rep.shape
    CH = np.asarray(RP.choices.T)                                 # (P, T)
    cond = RP.true_order
    W = len(RP.order_labels)
    labels, colors = RP.order_labels, RP.order_colors
    onehot = (cond[:, None] == np.arange(W)[None, :]).astype(np.float64)   # (P, W)
    swap = ppc['swap_mask']                                       # (P, T)
    bag = AN.bag_consistent_mask().transpose(1, 0, 2)            # (P, T, 4)
    err_obs = CH != 0
    err_rep = rep != 0
    acc = RP.accuracy
    print(f'{S} joint posterior samples, {P} participants, {T} trials')


    def pval(rep_stat, obs_stat):
        """P(T_rep >= T_obs) over the sample axis (axis 0 of rep_stat)."""
        return (rep_stat >= obs_stat[None]).mean(axis=0)


    def two_sided(p):
        return 2 * np.minimum(p, 1 - p)


    def pooled(x_p):
        """(..., P) per-participant statistic -> (..., W) condition sums."""
        return x_p @ onehot


    # ---------------------------------------------------------------- loglik ----
    ll_obs, ll_rep = ppc['ll_obs'].astype(np.float64), ppc['ll_rep'].astype(np.float64)   # (S, P)
    # T(y_obs, theta) varies over draws too (theta enters), so compare draw by draw
    p_ll = (ll_rep >= ll_obs).mean(0)                             # (P,)
    p_ll_w = (pooled(ll_rep) >= pooled(ll_obs)).mean(0)           # (W,)
    g_obs, g_rep = ll_obs.sum(1), ll_rep.sum(1)                   # (S,)
    p_ll_global = (g_rep >= g_obs).mean()
    print('\n[loglik] emission log-likelihood as test statistic, T(y, theta) = sum_t log p(y_t | B_t, O_t, lam)')
    print(f'  global: mean T(y_obs) {g_obs.mean():.1f}, mean T(y_rep) {g_rep.mean():.1f}, '
          f'mean difference {(g_obs - g_rep).mean():+.1f} (sd over draws {(g_obs - g_rep).std():.1f}), '
          f'p = P(T_rep >= T_obs) = {p_ll_global:.3f}')
    print(f'  {"cond":>5}  {"n":>3}  {"T_obs":>8}  {"T_rep":>8}  {"diff":>7}  {"p":>6}')
    for w in range(W):
        o, r = pooled(ll_obs)[:, w], pooled(ll_rep)[:, w]
        print(f'  {labels[w]:>5}  {int(onehot[:, w].sum()):>3}  {o.mean():>8.1f}  {r.mean():>8.1f}  '
              f'{(o - r).mean():>+7.1f}  {p_ll_w[w]:>6.3f}')
    print(f'  per participant: median p {np.median(p_ll):.3f}; p < 0.05: {(p_ll < 0.05).sum()}, '
          f'p > 0.95: {(p_ll > 0.95).sum()} of {P}')

    # ---------------------------------------------------------------- blocks ----
    nb = T // BLOCK
    blk_obs = err_obs[:, :nb * BLOCK].reshape(P, nb, BLOCK).sum(-1)              # (P, nb)
    blk_rep = err_rep[:, :, :nb * BLOCK].reshape(S, P, nb, BLOCK).sum(-1)        # (S, P, nb)
    bw_obs = np.einsum('pb,pw->wb', blk_obs, onehot)                              # (W, nb)
    bw_rep = np.einsum('spb,pw->swb', blk_rep, onehot)                            # (S, W, nb)
    p_blk = pval(bw_rep, bw_obs)                                                  # (W, nb)
    p_blk_all = pval(bw_rep.sum(1), bw_obs.sum(0))                                # (nb,)
    n_w = onehot.sum(0)
    print(f'\n[blocks] errors per block of {BLOCK} trials, summed over the participants of a '
          f'condition: p = P(T_rep >= T_obs) (< 0.5: fewer errors observed than replicated)')
    print(f'  {"cond":>5}  ' + '  '.join(f'{b * BLOCK:>3}-{(b + 1) * BLOCK - 1:<3}' for b in range(nb)))
    for w in range(W):
        print(f'  {labels[w]:>5}  ' + '  '.join(f'{p_blk[w, b]:>7.3f}' for b in range(nb)))
    print(f'  {"all":>5}  ' + '  '.join(f'{p_blk_all[b]:>7.3f}' for b in range(nb)))
    print(f'  extreme (two-sided p < 0.05) condition x block cells: '
          f'{(two_sided(p_blk) < 0.05).sum()}/{p_blk.size}')

    # ----------------------------------------------------------------- pairs ----
    def _pairs(err):
        return (err[..., :-1] & err[..., 1:]).sum(-1).astype(float)
    def _expected(err):
        n = err.sum(-1).astype(float)
        return n * (n - 1) / T
    pairs_obs, exp_obs = _pairs(err_obs), _expected(err_obs)                   # (P,)
    pairs_rep, exp_rep = _pairs(err_rep), _expected(err_rep)                   # (S, P)
    with np.errstate(invalid='ignore', divide='ignore'):
        idx_obs = np.where(exp_obs > 0, pairs_obs / exp_obs, np.nan)
        idx_rep = np.where(exp_rep > 0, pairs_rep / exp_rep, np.nan)
    ok_pairs = exp_obs > 0
    p_pairs = np.where(ok_pairs, np.nanmean(idx_rep >= idx_obs[None], axis=0), np.nan)
    g_idx_obs = pairs_obs.sum() / exp_obs.sum()
    g_idx_rep = pairs_rep.sum(1) / exp_rep.sum(1)                                # (S,)
    p_pairs_global = (g_idx_rep >= g_idx_obs).mean()
    w_idx_obs = pooled(pairs_obs) / pooled(exp_obs)
    w_idx_rep = pooled(pairs_rep) / pooled(exp_rep)
    p_pairs_w = (w_idx_rep >= w_idx_obs[None]).mean(0)
    print('\n[pairs] error clustering index = consecutive-error pairs / n(n-1)/T (1 = as many pairs '
          'as a random shuffle of the sequence)')
    print(f'  global: observed {g_idx_obs:.3f} ({int(pairs_obs.sum())} pairs), replicated '
          f'{g_idx_rep.mean():.3f} [{np.percentile(g_idx_rep, 2.5):.3f}, {np.percentile(g_idx_rep, 97.5):.3f}] '
          f'({pairs_rep.sum(1).mean():.0f} pairs), p = {p_pairs_global:.3f}')
    print('  per condition obs / rep / p: ' + ', '.join(
        f'{labels[w]} {w_idx_obs[w]:.2f}/{w_idx_rep[:, w].mean():.2f}/{p_pairs_w[w]:.3f}' for w in range(W)))
    print(f'  per participant ({ok_pairs.sum()} with >= 2 errors): median p {np.nanmedian(p_pairs):.3f}; '
          f'p < 0.05: {np.nansum(p_pairs < 0.05)}, p > 0.95: {np.nansum(p_pairs > 0.95)}')

    # ------------------------------------------------------------- swapshare ----
    chosen_bag_obs = np.take_along_axis(bag, CH[..., None], -1)[..., 0]           # (P, T)
    chosen_bag_rep = np.take_along_axis(np.broadcast_to(bag, (S, P, T, 4)),
                                        rep[..., None].astype(np.int64), -1)[..., 0]
    num_obs = pooled((err_obs & swap & chosen_bag_obs).sum(1).astype(float))      # (W,)
    den_obs = pooled((err_obs & swap).sum(1).astype(float))
    num_rep = pooled((err_rep & swap[None] & chosen_bag_rep).sum(2).astype(float))  # (S, W)
    den_rep = pooled((err_rep & swap[None]).sum(2).astype(float))
    share_obs = num_obs / np.maximum(den_obs, 1)
    share_rep = num_rep / np.maximum(den_rep, 1)
    p_share = pval(share_rep, share_obs)
    p_share_global = (num_rep.sum(1) / den_rep.sum(1) >= num_obs.sum() / den_obs.sum()).mean()
    print('\n[swapshare] on role-swap trials, share of errors that pick the role-swapped '
          '(utterance-consistent) scene')
    print(f'  global: observed {num_obs.sum() / den_obs.sum():.3f} ({int(num_obs.sum())}/{int(den_obs.sum())}), '
          f'replicated {(num_rep.sum(1) / den_rep.sum(1)).mean():.3f}, p = {p_share_global:.3f}')
    print(f'  {"cond":>5}  {"obs share":>9}  {"n err":>5}  {"rep share":>9}  {"p":>6}')
    for w in range(W):
        print(f'  {labels[w]:>5}  {share_obs[w]:>9.3f}  {int(den_obs[w]):>5}  '
              f'{share_rep[:, w].mean():>9.3f}  {p_share[w]:>6.3f}')

    # ------------------------------------------------ by survivor count ----
    nS = AN.unpack_choices(ppc['n_surv_packed'], T).astype(np.int8) + 1     # (S, P, T)
    def _split(mask):
        n_obs = pooled((err_obs[None] & mask).sum(2).astype(float))       # (S, W)
        n_rep = pooled((err_rep & mask).sum(2).astype(float))
        n_tr = pooled(mask.sum(2).astype(float))
        return n_obs, n_rep, n_tr
    SURV_CLASSES = [(1, 'nS = 1'), (2, 'nS = 2'), (3, 'nS = 3'), (4, 'nS = 4 (uniform)')]
    print('\n[survivors] error rate by the number of candidates the sampled state leaves tied '
          '(share of trials; obs / rep error rate averaged over draws; p = P(T_rep >= T_obs) '
          'on the error count, draw by draw)')
    print(f'  {"cond":>5}  ' + '  '.join(f'{lab:>22}' for _, lab in SURV_CLASSES))
    split = {}                                                            # (w, k) -> (obs, rep, p, n)
    for w in range(W):
        cells = []
        for k, _ in SURV_CLASSES:
            n_obs, n_rep, n_tr = _split(nS == k)
            c = ((n_obs[:, w] / n_tr[:, w]).mean(), (n_rep[:, w] / n_tr[:, w]).mean(),
                 (n_rep[:, w] >= n_obs[:, w]).mean(), n_tr[:, w].mean())
            split[w, k] = c
            cells.append(f'{c[3] / (T * n_w[w]):.2f} {c[0]:.3f}/{c[1]:.3f} {c[2]:.3f}'
                         if c[3] > 0 else f'{0:.2f} {"(no trials)":>17}')
        print(f'  {labels[w]:>5}  ' + '  '.join(f'{c:>22}' for c in cells))
    cells, excess = [], []
    for k, _ in SURV_CLASSES:
        n_obs, n_rep, n_tr = _split(nS == k)
        c = ((n_obs.sum(1) / n_tr.sum(1)).mean(), (n_rep.sum(1) / n_tr.sum(1)).mean(),
             (n_rep.sum(1) >= n_obs.sum(1)).mean(), n_tr.sum(1).mean())
        cells.append(f'{c[3] / (T * P):.2f} {c[0]:.3f}/{c[1]:.3f} {c[2]:.3f}'
                     if c[3] > 0 else f'{0:.2f} {"(no trials)":>17}')
        excess.append((n_rep.sum(1) - n_obs.sum(1)).mean())
    print(f'  {"all":>5}  ' + '  '.join(f'{c:>22}' for c in cells))
    print('  excess replicated errors per draw, all conditions: '
          + ', '.join(f'{lab} {e:+.0f}' for (_, lab), e in zip(SURV_CLASSES, excess)))

    # ---------------------------------------------------------- calibration ----
    stats_p = {'loglik': p_ll, 'total errors': ppc['p_total'], 'swap errors': ppc['p_swap'],
               'error clustering': p_pairs[ok_pairs]}
    print('\n[calibration] per-participant p-values (fraction < 0.05 / > 0.95; 0.05 each if calibrated;'
          ' a mixed PPC is conservative, so fewer extremes are expected under a correct model)')
    for k, v in stats_p.items():
        print(f'  {k:>13}: median {np.median(v):.3f}, < 0.05: {(v < 0.05).mean():.3f}, '
              f'> 0.95: {(v > 0.95).mean():.3f}')

    # ---------------------------------------------------------------- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.scatter(g_obs, g_rep, s=6, alpha=0.4, c='#377eb8')
    lim = [min(g_obs.min(), g_rep.min()), max(g_obs.max(), g_rep.max())]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel(r'$T(y_{obs}, \theta^{(s)})$: emission log-lik of the data')
    ax.set_ylabel(r'$T(y_{rep}^{(s)}, \theta^{(s)})$: of the replicate')
    ax.set_title(f'Global log-likelihood discrepancy, p = {p_ll_global:.3f}', fontsize=10)
    ax = axes[0, 1]
    ax.hist(p_ll, bins=20, range=(0, 1), color='#377eb8', alpha=0.85)
    ax.axvline(0.05, color='r', ls='--', lw=0.8); ax.axvline(0.95, color='r', ls='--', lw=0.8)
    ax.set_xlabel('per-participant p (log-likelihood)'); ax.set_ylabel('# participants')
    ax.set_title(f'Per-participant: {(p_ll < 0.05).sum()} below 0.05, {(p_ll > 0.95).sum()} above 0.95',
                 fontsize=10)
    ax = axes[1, 0]
    AN.violins(ax, [pooled(ll_obs)[:, w] - pooled(ll_rep)[:, w] for w in range(W)], labels, colors)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    for w in range(W):
        ax.text(w, ax.get_ylim()[1], f'p={p_ll_w[w]:.2f}', ha='center', va='top', fontsize=8)
    ax.set_ylabel(r'$T(y_{obs}) - T(y_{rep})$ summed over the condition')
    ax.set_title('Per condition (> 0: data more probable than replicates)', fontsize=10)
    ax = axes[1, 1]
    ax.scatter(acc, p_ll, s=8, alpha=0.5, c=[colors[w] for w in cond])
    ax.axhline(0.05, color='r', ls='--', lw=0.8); ax.axhline(0.95, color='r', ls='--', lw=0.8)
    ax.set_xlabel('participant accuracy'); ax.set_ylabel('per-participant p (log-likelihood)')
    ax.set_title('Where does the misfit sit?', fontsize=10)
    plt.tight_layout()
    IO.savefig(fig, 'ppc_loglik.png'); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    xb = np.arange(nb) * BLOCK + BLOCK / 2
    for w in range(W):
        ax = axes.flat[w]
        rate_rep = bw_rep[:, w] / (n_w[w] * BLOCK)
        lo, hi = np.percentile(rate_rep, [2.5, 97.5], axis=0)
        ax.fill_between(xb, lo, hi, color=colors[w], alpha=0.25, label='replicates 95%')
        ax.plot(xb, np.median(rate_rep, 0), color=colors[w], lw=1.2, label='replicate median')
        ax.plot(xb, bw_obs[w] / (n_w[w] * BLOCK), 'ko-', ms=4, lw=1.0, label='observed')
        for b in range(nb):
            if two_sided(p_blk[w, b]) < 0.05:
                ax.text(xb[b], hi[b], '*', ha='center', va='bottom', fontsize=11, color='r')
        ax.set_title(f'{labels[w]} (n={int(n_w[w])})', fontsize=10)
        ax.set_xlabel('trial'); ax.set_ylim(0, 0.8)
        if w == 0:
            ax.legend(fontsize=7); ax.set_ylabel('error rate per block')
    ax = axes.flat[W]
    rate_rep = bw_rep.sum(1) / (P * BLOCK)
    lo, hi = np.percentile(rate_rep, [2.5, 97.5], axis=0)
    ax.fill_between(xb, lo, hi, color='#777', alpha=0.25)
    ax.plot(xb, np.median(rate_rep, 0), color='#777', lw=1.2)
    ax.plot(xb, bw_obs.sum(0) / (P * BLOCK), 'ko-', ms=4, lw=1.0)
    for b in range(nb):
        if two_sided(p_blk_all[b]) < 0.05:
            ax.text(xb[b], hi[b], '*', ha='center', va='bottom', fontsize=11, color='r')
    ax.set_title('all conditions', fontsize=10); ax.set_xlabel('trial'); ax.set_ylim(0, 0.8)
    ax = axes.flat[W + 1]
    im = ax.imshow(p_blk, aspect='auto', cmap='RdBu', vmin=0, vmax=1)
    ax.set_yticks(range(W)); ax.set_yticklabels(labels)
    ax.set_xticks(range(nb)); ax.set_xticklabels([f'{b * BLOCK}' for b in range(nb)], fontsize=8)
    ax.set_xlabel('block start trial'); ax.set_title('p = P(T_rep >= T_obs) per cell', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Errors per {BLOCK}-trial block: observed vs mixed posterior predictive '
                 f'(* two-sided p < 0.05)', fontsize=11)
    plt.tight_layout()
    IO.savefig(fig, 'ppc_blocks.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    ax = axes[3]
    xw = np.arange(W)
    cols = ['#377eb8', '#4daf4a', '#ff7f00', '#999999']
    for j, ((k, lab), col) in enumerate(zip(SURV_CLASSES, cols)):
        off = (j - 1.5) * 0.2
        ax.bar(xw + off, [split[w, k][1] for w in range(W)], width=0.19, color=col, alpha=0.5,
               label=f'{lab}: replicated')
        ax.plot(xw + off, [split[w, k][0] for w in range(W)], 'k_', ms=7, mew=2,
                label='observed' if j == 0 else None)
    ax.set_xticks(xw); ax.set_xticklabels(labels)
    ax.set_ylabel('error rate'); ax.set_ylim(0, 0.9)
    ax.set_title('Errors by # candidates tied in the sampled state', fontsize=10); ax.legend(fontsize=7)
    ax = axes[0]
    ax.hist(g_idx_rep, bins=30, color='#984ea3', alpha=0.8, label='replicates')
    ax.axvline(g_idx_obs, color='k', lw=1.5, label='observed')
    ax.set_xlabel('error clustering index, all participants'); ax.set_ylabel('# posterior draws')
    ax.set_title(f'Error clustering, p = {p_pairs_global:.3f}', fontsize=10); ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(p_pairs[ok_pairs], bins=20, range=(0, 1), color='#984ea3', alpha=0.85)
    ax.axvline(0.05, color='r', ls='--', lw=0.8); ax.axvline(0.95, color='r', ls='--', lw=0.8)
    ax.set_xlabel('per-participant p (error clustering)'); ax.set_ylabel('# participants')
    ax.set_title(f'{np.nansum(p_pairs < 0.05)} participants with more clustering than replicated',
                 fontsize=10)
    ax = axes[2]
    AN.violins(ax, [share_rep[:, w] for w in range(W)], labels, colors)
    ax.plot(range(W), share_obs, 'k_', ms=18, mew=2, label='observed')
    for w in range(W):
        ax.text(w, 1.02, f'p={p_share[w]:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel('share of swap-trial errors on the role-swapped scene')
    ax.set_title(f'Structural errors on swap trials (global p = {p_share_global:.3f})', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    IO.savefig(fig, 'ppc_sequential.png'); plt.close(fig)

    fig, axes = plt.subplots(1, len(stats_p), figsize=(4 * len(stats_p), 3.6))
    for ax, (k, v) in zip(axes, stats_p.items()):
        ax.hist(v, bins=20, range=(0, 1), color='#4daf4a', alpha=0.85)
        ax.axhline(P / 20, color='k', ls=':', lw=0.8)
        ax.set_title(f'{k}: {(v < 0.05).sum()} < 0.05, {(v > 0.95).sum()} > 0.95', fontsize=9)
        ax.set_xlabel('per-participant p')
    axes[0].set_ylabel('# participants (dotted: uniform)')
    fig.suptitle('Per-participant posterior predictive p-values by test statistic', fontsize=10)
    plt.tight_layout()
    IO.savefig(fig, 'ppc_pvalue_calibration.png'); plt.close(fig)
    print('\ndone')


if __name__ == '__main__':
    diagnostics()
    pvalues()
