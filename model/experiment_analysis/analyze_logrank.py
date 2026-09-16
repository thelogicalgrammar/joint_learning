"""
Secondary cross-check: non-parametric K-sample log-rank test + Kaplan-Meier
curves across the 6 WO conditions, on the latent time-to-true-order.

Per-participant event time = posterior median of the first trial where the
sampled O equals the participant's TRUE order (or T if never); censored iff
never reached. See README ("Secondary (survival) analyses") for caveats.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import (multivariate_logrank_test,
                                   pairwise_logrank_test)
import data as RP
import helpers as HLP

order_labels = RP.order_labels
order_colors = RP.order_colors
true_order = RP.true_order

st = HLP.load_derived('per_participant_settle.pkl')
settle_sp, T = st['settle'], st['T']                            # (S, P)
df = HLP.km_frame(settle_sp, true_order, T)
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
HLP.savefig(fig, 'logrank.png')
