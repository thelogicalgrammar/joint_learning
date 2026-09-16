"""
Posterior predictive p-values for the hierarchical fit.

Reads the replicate choice sequences and emission log-likelihoods that
infer_per_participant.py stores per joint posterior sample (B, O, tpar)
and compares test statistics T(y_rep, theta) with T(y_obs, theta):

    p = P( T(y_rep, theta) >= T(y_obs, theta) )   over posterior draws theta

with theta = (B, O, tpar) the sampled latent trajectories and parameters.
This is the mixed (conditional) posterior predictive check: replicates are
generated given the latent trajectories inferred from the observed choices,
so it tests the emission model (lapse mixture, survivor rule) and the
sequential structure of the errors, not the transition prior. The check is
conservative — the latents have already absorbed part of any misfit — so
per-participant p-values concentrate around 0.5 even under a correct model,
and a p-value near 0 or 1 is a strong signal.

Test statistics
  loglik      the emission log-likelihood sum_t log p(y_t | B_t, O_t, lam):
              per participant, per condition (sum over its participants)
              and global. (The transition terms of the full log-likelihood
              depend on the latents only and cancel between y_obs and y_rep.)
              A small p-value says the observed choices are less probable
              under the fitted state than the model's own replicates.
  blocks      number of errors per block of 25 trials, summed over the
              participants of a condition: is the learning-curve shape
              reproduced, and where in the session does the misfit sit?
  pairs       clustering of errors: the number of consecutive-error pairs
              (t, t+1) divided by its expectation under a random shuffle of
              the participant's own sequence, n(n-1)/T for n errors (pooled:
              sum of pairs / sum of expectations). Lapses are independent in
              the model; clustered attention lapses would give a larger
              index than replicated. The index is invariant to the error
              count, so this does not merely re-test the total error rate.
  swapshare   on role-swap trials, the share of errors that pick a candidate
              consistent with the utterance up to word order (the role-swapped
              scene) rather than an unrelated one, per condition: are errors
              structural (order confusion) as the model claims?
  survivors   errors split by the number nS of candidate scenes the sampled
              state leaves tied at the top: nS = 1 (unique prediction: an
              error is a lapse), nS = 2, 3 (an error on another survivor costs
              nothing in likelihood — the model is indifferent among them)
              and nS = 4 (uniform: no committed heard word, no or all
              survivors). The log-likelihood only sees lapses, so this split
              says where an error-rate misfit sits: on the tied / uniform
              trials it means participants know more than the sampled
              lexicon expresses.
  total/swap  the error counts of infer_per_participant.py (reported here
              alongside for the calibration summary).

Reads:  results/per_participant_ppc.pkl
Writes: figures/ppc_loglik.png, ppc_blocks.png, ppc_sequential.png,
        figures/ppc_pvalue_calibration.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data as RP
import helpers as HLP

BLOCK = 25
ppc = HLP.load_derived('per_participant_ppc.pkl')
if 'rep_choice_packed' not in ppc:
    raise SystemExit('per_participant_ppc.pkl has no replicate choice sequences: '
                     're-run infer_per_participant.py')
T = int(ppc['T'])
rep = HLP.unpack_choices(ppc['rep_choice_packed'], T)        # (S, P, T) uint8
S, P, _ = rep.shape
CH = np.asarray(RP.choices.T)                                 # (P, T)
cond = RP.true_order
W = len(RP.order_labels)
labels, colors = RP.order_labels, RP.order_colors
onehot = (cond[:, None] == np.arange(W)[None, :]).astype(np.float64)   # (P, W)
swap = ppc['swap_mask']                                       # (P, T)
bag = HLP.bag_consistent_mask().transpose(1, 0, 2)            # (P, T, 4)
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
nS = HLP.unpack_choices(ppc['n_surv_packed'], T).astype(np.int8) + 1     # (S, P, T)
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
HLP.violins(ax, [pooled(ll_obs)[:, w] - pooled(ll_rep)[:, w] for w in range(W)], labels, colors)
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
HLP.savefig(fig, 'ppc_loglik.png'); plt.close(fig)

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
HLP.savefig(fig, 'ppc_blocks.png'); plt.close(fig)

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
HLP.violins(ax, [share_rep[:, w] for w in range(W)], labels, colors)
ax.plot(range(W), share_obs, 'k_', ms=18, mew=2, label='observed')
for w in range(W):
    ax.text(w, 1.02, f'p={p_share[w]:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, 1.1); ax.set_ylabel('share of swap-trial errors on the role-swapped scene')
ax.set_title(f'Structural errors on swap trials (global p = {p_share_global:.3f})', fontsize=10)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
HLP.savefig(fig, 'ppc_sequential.png'); plt.close(fig)

fig, axes = plt.subplots(1, len(stats_p), figsize=(4 * len(stats_p), 3.6))
for ax, (k, v) in zip(axes, stats_p.items()):
    ax.hist(v, bins=20, range=(0, 1), color='#4daf4a', alpha=0.85)
    ax.axhline(P / 20, color='k', ls=':', lw=0.8)
    ax.set_title(f'{k}: {(v < 0.05).sum()} < 0.05, {(v > 0.95).sum()} > 0.95', fontsize=9)
    ax.set_xlabel('per-participant p')
axes[0].set_ylabel('# participants (dotted: uniform)')
fig.suptitle('Per-participant posterior predictive p-values by test statistic', fontsize=10)
plt.tight_layout()
HLP.savefig(fig, 'ppc_pvalue_calibration.png'); plt.close(fig)
print('\ndone')
