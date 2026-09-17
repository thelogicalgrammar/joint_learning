"""
Step 7 — secondary survival cross-checks on the latent first-passage time of the
order state to the true order (results/per_participant_settle.pkl).

logrank(): Kaplan-Meier curves and log-rank tests by condition (logrank.png).
weibull(): per-posterior-sample Weibull fits (weibull.png, ~5 min).

Kept as a model-external cross-check of 03_condition_effects.py; see the caveats
in README.md.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import (multivariate_logrank_test,
                                   pairwise_logrank_test)
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
import time
import warnings
from lifelines import KaplanMeierFitter, WeibullFitter
from lifelines.exceptions import ConvergenceError


def logrank():
    order_labels = RP.order_labels
    order_colors = RP.order_colors
    true_order = RP.true_order

    st = IO.load_derived('per_participant_settle.pkl')
    settle_sp, T = st['settle'], st['T']                            # (S, P)
    df = AN.km_frame(settle_sp, true_order, T)
    n_event = int(df['event'].sum())
    print(f'{len(df)} participants; {n_event} reached (event=1), '
          f'{len(df) - n_event} censored at T={T}.')

    # K-sample log-rank
    res = multivariate_logrank_test(df['T'], df['group'], df['event'])
    print('\n=== K-sample log-rank test ===')
    print(f'  chi^2 = {res.test_statistic:.2f}  df = {res.degrees_of_freedom}  '
          f'p = {res.p_value:.3e}')

    # Pairwise with Holm-Bonferroni
    pw = pairwise_logrank_test(df['T'], df['group'], df['event'],
                                p_adjust_method='holm')
    print('\n=== Pairwise log-rank (Holm-Bonferroni adjusted) ===')
    print(pw.summary[['test_statistic', 'p', '-log2(p)']].round(3))

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for w, lbl in enumerate(order_labels):
        m = df['group'] == lbl
        kmf = KaplanMeierFitter(label=lbl)
        kmf.fit(df.loc[m, 'T'], df.loc[m, 'event'])
        kmf.plot_cumulative_density(ax=ax, color=order_colors[w], lw=1.7,
                                     ci_show=False)
    ax.axvline(T, color='#888', ls='--', lw=0.7)
    ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
    ax.set_xlabel('trial'); ax.set_ylabel('cumulative P(learned)')
    ax.set_title('Kaplan-Meier learning curves')
    ax.legend(fontsize=9, loc='lower right')

    ax = axes[1]
    M = np.full((6, 6), np.nan)
    for _, row in pw.summary.reset_index().iterrows():
        i = order_labels.index(row['level_0']); j = order_labels.index(row['level_1'])
        M[i, j] = M[j, i] = row['p']
    np.fill_diagonal(M, 1.0)
    log_M = -np.log10(np.where(np.isnan(M), 1, M))
    im = ax.imshow(log_M, cmap='viridis', vmin=0, vmax=10)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(order_labels); ax.set_yticklabels(order_labels)
    ax.set_title('-log10(Holm-adjusted pairwise log-rank p)')
    for i in range(6):
        for j in range(6):
            if i != j and not np.isnan(M[i, j]):
                t = f'{M[i, j]:.0e}' if M[i, j] < 1e-3 else f'{M[i, j]:.2f}'
                ax.text(j, i, t, ha='center', va='center', fontsize=7,
                        color='white' if log_M[i, j] > 4 else 'black')
    plt.colorbar(im, ax=ax, label='-log10(p)')
    plt.tight_layout()
    IO.savefig(fig, 'logrank.png')


def weibull():
    order_labels = RP.order_labels
    order_colors = RP.order_colors
    true_order = RP.true_order

    # settle[s, p] = first t where O = true (or T if never); reached[s, p] = bool
    st = IO.load_derived('per_participant_settle.pkl')
    settle, reached, T = st['settle'], st['reached'], st['T']
    S = settle.shape[0]

    # Per (sample, WO): WeibullFitter with right-censoring
    print('Fitting Weibull per (sample, WO) with lifelines...')
    t0 = time.time()
    shape_arr = np.full((6, S), np.nan, dtype=np.float32)
    median_arr = np.full((6, S), np.nan, dtype=np.float32)
    wf = WeibullFitter()
    n_fail = 0
    for w in range(6):
        pid_w = np.where(true_order == w)[0]
        times = settle[:, pid_w].astype(float)
        events = reached[:, pid_w].astype(int)
        for s in range(S):
            if events[s].sum() == 0:            # nothing observed: no MLE
                n_fail += 1
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    wf.fit(times[s], events[s])
            except ConvergenceError:
                n_fail += 1
                continue
            median_arr[w, s] = wf.median_survival_time_
            shape_arr[w, s] = wf.rho_
        print(f'  {order_labels[w]}: done ({time.time() - t0:.0f}s)', flush=True)
    if n_fail:
        print(f'  {n_fail} (sample, WO) fits skipped (no events or no convergence)')


    def summary(x, digits=1):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 'n/a (no converged Weibull fits)'
        lo, hi = np.percentile(x, [2.5, 97.5])
        return f'{np.median(x):.{digits}f}  [{lo:.{digits}f}, {hi:.{digits}f}]'


    print('\n=== Weibull median time-to-learn per WO (posterior median [95% CI]) ===')
    for w in range(6):
        print(f'  {order_labels[w]:>3}  (n={int((true_order == w).sum()):>3})  {summary(median_arr[w])}')

    print('\n=== Weibull shape k per WO (k=1 => exponential) ===')
    for w in range(6):
        print(f'  {order_labels[w]:>3}: k = {summary(shape_arr[w], 3)}')

    print('\n=== Median time ratio vs SVO (paired across posterior samples) ===')
    for w in range(1, 6):
        r = median_arr[w] / median_arr[0]
        r = r[np.isfinite(r)]
        print(f'  {order_labels[w]} / SVO: {summary(r, 2)}x   P(slower) = {(r > 1).mean():.2f}')

    # --- plot: KM (per-ppt median + reached) overlaid with Weibull at posterior median params
    df = AN.km_frame(settle, true_order, T)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ts = np.arange(0, 500)
    for w, lbl in enumerate(order_labels):
        m = df['group'] == lbl
        kmf = KaplanMeierFitter(label=f'{lbl} (KM)')
        kmf.fit(df.loc[m, 'T'], df.loc[m, 'event'])
        kmf.plot_cumulative_density(ax=ax, color=order_colors[w], lw=1.6, ci_show=False)
        # Weibull at posterior median params: S(t) = exp(-(t/scale)^shape)
        # median = scale * (ln 2)^(1/shape)  =>  scale = median / (ln 2)^(1/shape)
        if np.isfinite(median_arr[w]).any():
            med = float(np.nanmedian(median_arr[w])); k = float(np.nanmedian(shape_arr[w]))
            scale = med / (np.log(2.0) ** (1.0 / k))
            ax.plot(ts, 1 - np.exp(-(ts / scale) ** k), color=order_colors[w],
                    lw=1.0, ls='--', alpha=0.8)
    ax.axvline(T, color='#888', ls=':', lw=0.7)
    ax.set_xlim(0, 500); ax.set_ylim(0, 1.02)
    ax.set_xlabel('trial'); ax.set_ylabel('cumulative P(learned)')
    ax.set_title('KM (solid) vs Weibull (dashed) per WO')
    ax.legend(fontsize=7, loc='lower right', ncol=2)

    ax = axes[1]
    AN.violins(ax, [median_arr[w][np.isfinite(median_arr[w])] for w in range(6)],
                order_labels, order_colors)
    ax.set_ylabel('Weibull median time-to-learn (trials)')
    ax.set_title('Weibull median per WO (posterior)')
    ax.axhline(T, color='#888', ls='--', lw=0.7)
    plt.tight_layout()
    IO.savefig(fig, 'weibull.png')


if __name__ == '__main__':
    logrank()
    weibull()
