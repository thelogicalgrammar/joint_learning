"""
Secondary cross-check: parametric (Weibull) time-to-true-order with
right-censoring at T, fit per (posterior sample, word-order condition)
using `lifelines.WeibullFitter`.

Yields a posterior over the Weibull median time-to-learn and the shape
parameter k per WO. KM curves overlaid with the fitted Weibull as a sanity
check. See README ("Secondary (survival) analyses") for caveats — in
particular, medians above T are extrapolations, and k < 1 is expected from
between-participant heterogeneity in a constant per-participant hazard.
"""
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, WeibullFitter
from lifelines.exceptions import ConvergenceError
import data as RP
import helpers as HLP

order_labels = RP.order_labels
order_colors = RP.order_colors
true_order = RP.true_order

# settle[s, p] = first t where O = true (or T if never); reached[s, p] = bool
st = HLP.load_derived('per_participant_settle.pkl')
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
df = HLP.km_frame(settle, true_order, T)
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
HLP.violins(ax, [median_arr[w][np.isfinite(median_arr[w])] for w in range(6)],
            order_labels, order_colors)
ax.set_ylabel('Weibull median time-to-learn (trials)')
ax.set_title('Weibull median per WO (posterior)')
ax.axhline(T, color='#888', ls='--', lw=0.7)
plt.tight_layout()
HLP.savefig(fig, 'weibull.png')
