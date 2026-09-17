"""
Condition-level learnability from the hierarchical fit.

The fit puts a condition-level hyperprior on the per-participant
parameters: tpar[p, k] ~ N(mu[w(p), k], sigma_k^2) with w(p) the word-order
condition. This script summarises the posterior of the condition means
mu[w, k] on the natural scale, the between-condition sd tau_k, and the
contrasts vs SVO — all within the model, no post-hoc test on latent
first-passage times.

Reads:  results/hierarchical_fit.pkl
Writes: figures/condition_effects.png
Prints: per-condition posterior medians + 95% CIs, contrasts vs SVO and
        by typological group (S- / V- / O-initial), tau posteriors.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm import model as NB
from jointlearn.hmm import config as CFG

fit = IO.load_fit()
labels = fit['cond_labels']
colors = RP.order_colors
W, K = len(labels), fit['n_tpar']

mu = IO.pool_hyper(fit, 'mu')            # (S, W, K)
tau = IO.pool_hyper(fit, 'tau')          # (S, K)
sigma = IO.pool_hyper(fit, 'sigma')      # (S, K)
print(f'{len(fit["chains"])} chains, {mu.shape[0]} pooled hyperposterior '
      f'samples, {W} conditions, {K} fitted params')

# Display (label, log-scale y axis) per fitted coordinate, keyed by its
# config.PARAM_NAMES entry; the transformed -> natural mapping itself comes
# from model.tpar_to_nat_np.
DISPLAY = {'logit_lam': ('lambda (lapse rate)', False),
           'log_kappa': ('kappa (single-move log-cost)', False),
           'logit_eps_o': ('eps_o (order-switch rate / trial)', True),
           'logit_p_commit_o': ('p_commit_o (order-commit rate / trial)', True)}
NAT = [DISPLAY[name] for name in CFG.PARAM_NAMES]
I_COMMIT = CFG.PARAM_NAMES.index('logit_p_commit_o')


def ci(x, q=(2.5, 97.5)):
    return np.percentile(x, q)


def fmt(x, digits=4):
    lo, hi = ci(x)
    return f'{np.median(x):.{digits}g}  [{lo:.{digits}g}, {hi:.{digits}g}]'


def ratio_line(name, r):
    return f'    {name}: {np.median(r):5.2f}  [{ci(r)[0]:.2f}, {ci(r)[1]:.2f}]   P(>1) = {(r > 1).mean():.3f}'


def geometric_median_time(p):
    """Median number of trials until a geometric(p) event."""
    return np.log(0.5) / np.log1p(-p)


nat_all = NB.tpar_to_nat_np(mu)                                      # (S, W, K)
nat = [nat_all[:, :, k] for k in range(K)]                            # each (S, W)

for k in range(K):
    x = nat[k]
    print(f'\n=== {NAT[k][0]} — condition means (posterior median [95% CI]) ===')
    for w in range(W):
        extra = ''
        if k == I_COMMIT:
            extra = f'   median trials-to-commit {fmt(geometric_median_time(x[:, w]), 3)}'
        print(f'  {labels[w]:>4}: {fmt(x[:, w])}{extra}')
    print(f'  between-condition sd tau (transformed scale): {fmt(tau[:, k], 3)}'
          f'   (within-condition sigma {np.median(sigma[:, k]):.3f})')
    print(f'  contrasts vs {labels[0]} (ratio of condition means):')
    for w in range(1, W):
        print(ratio_line(f'{labels[w]:>4} / {labels[0]}', x[:, w] / x[:, 0]))
    gm = {g: x[:, idx].mean(axis=1) for g, idx in RP.TYPOLOGY.items()}
    print('  typological groups (mean of the two condition means):')
    for g in gm:
        print(f'    {g:>9}: {fmt(gm[g])}')
    for a, b in [('S-initial', 'V-initial'), ('V-initial', 'O-initial'),
                 ('S-initial', 'O-initial')]:
        print(ratio_line(f'{a} / {b}', gm[a] / gm[b]))

# ---- figure: posterior of condition means per param + trials-to-commit ----
fig, axes = plt.subplots(1, K + 1, figsize=(3.6 * (K + 1), 4.2))
for k in range(K):
    ax = axes[k]
    AN.violins(ax, [nat[k][:, w] for w in range(W)], labels, colors)
    ax.set_title(NAT[k][0], fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    if NAT[k][1]:
        ax.set_yscale('log')
axes[0].set_ylabel('condition mean (natural scale)')
ax = axes[K]
mt = geometric_median_time(nat[I_COMMIT])                              # (S, W)
AN.violins(ax, [mt[:, w] for w in range(W)], labels, colors)
ax.axhline(RP.T, color='#888', ls='--', lw=0.7)
ax.set_yscale('log')
ax.tick_params(axis='x', labelsize=8)
ax.set_title('median trials-to-commit (from p_commit_o)', fontsize=9)
plt.tight_layout()
IO.savefig(fig, 'condition_effects.png')
