"""
Results, section "Posterior predictive checks" (main text).

Everything this section of the paper reports is computed here, in the order
it appears in the paper:

  Table    test statistics pooled over all participants: observed vs
           replicated value and the posterior predictive p-value, for the
           choice log-likelihood, the error rate, error clustering, and the
           share of role-swap errors
  Text     per-participant log-likelihood p-values; how many condition x
           block cells have fewer observed errors than nearly all replicates
  Figure   (a) error rate per 25-trial block, observed vs replicated;
           (b) error rate by the number of candidates the sampled state
           leaves tied, observed vs replicated; (c) observed - replicated
           error rate per word order (all trials; 2 and 4 tied), with a
           table of the same numbers
  Table    errors by number of tied candidates: share of trials, observed
           and replicated error rate, excess replicated errors

The check is a *mixed* posterior predictive check: for each joint posterior
draw theta = (B, O, parameters) one replicate choice sequence y_rep is drawn
from the choice model given the sampled latent trajectories
(02_postprocess.py). It therefore tests the choice model and the error
structure given the latent states, not the transition prior, and is
conservative (the latents have already absorbed some misfit). For a test
statistic T, p = P(T(y_rep, theta) >= T(y_obs, theta)) over draws; p near 1
means the model predicts more of T than observed, p near 0 less.

Statistics:
  log-likelihood    sum over trials of log P(y_t | B_t, O_t, lambda)
  error rate        share of trials on which the chosen picture is wrong
  error clustering  consecutive error pairs / n(n-1)/T, summed over
                    participants (1 = as many pairs as a random shuffle)
  role-swap share   on role-swap trials (>= 2 candidates match the
                    utterance up to word order, ~9% of trials), the share of
                    errors that pick the role-swapped picture

Reads:  results/per_participant_ppc.pkl
Writes: figures/ppc.pdf (+ .png)
        results/05_ppc.md   report: the numbers as Markdown tables and LaTeX
                            to paste into the paper
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm.report import num, summ, fmt_p, p_tex, Report

BLOCK = 25                                           # trials per block in the figure
TIES = [1, 2, 4]                                     # candidates left tied (3 never occurs)

ppc = IO.load_derived('per_participant_ppc.pkl')
T = int(ppc['T'])
rep = AN.unpack_choices(ppc['rep_choice_packed'], T)          # (S, P, T) replicate choices
S, P, _ = rep.shape
CH = np.asarray(RP.choices.T)                                 # (P, T), 0 = correct picture
cond = np.asarray(RP.true_order)
labels = RP.order_labels
W = len(labels)
err_obs = CH != 0                                             # (P, T)
err_rep = rep != 0                                            # (S, P, T)

R = Report()
out = R.out
out('# Results: posterior predictive checks')
out()
out(f'Source: `05_ppc.py` on `per_participant_ppc.pkl` ({S} joint posterior draws, '
    f'{P} participants, {T} trials). p = P(T(y_rep) >= T(y_obs)) over draws; '
    'p near 1: the model predicts more than observed.')
out()


# ---------------------------------------------------------------- table: test statistics ----

# choice log-likelihood (depends on theta, so compared draw by draw)
ll_obs = ppc['ll_obs'].astype(np.float64).sum(1)              # (S,)
ll_rep = ppc['ll_rep'].astype(np.float64).sum(1)


def clustering(err):
    """Error clustering index of error sequences (..., P, T), pooled over P."""
    n = err.sum(-1).astype(float)
    pairs = (err[..., :-1] & err[..., 1:]).sum(-1)
    return pairs.sum(-1) / (n * (n - 1) / T).sum(-1)


# role-swap share: errors on role-swap trials that pick the role-swapped picture
swap = ppc['swap_mask']                                       # (P, T)
bag = AN.bag_consistent_mask().transpose(1, 0, 2)             # (P, T, 4)
on_swapped_obs = np.take_along_axis(bag, CH[..., None], -1)[..., 0]
on_swapped_rep = np.take_along_axis(np.broadcast_to(bag, (S, P, T, 4)),
                                    rep[..., None].astype(np.int64), -1)[..., 0]
del rep

STATS = [  # name, observed (scalar or per draw), replicated per draw, number format
    ('Choice log-likelihood', ll_obs, ll_rep, 'loglik'),
    ('Error rate', err_obs.mean(), err_rep.mean(axis=(1, 2)), 'rate'),
    ('Error clustering index', clustering(err_obs), clustering(err_rep), 'index'),
    ('Role-swap errors on the swapped picture',
     (err_obs & swap & on_swapped_obs).sum() / (err_obs & swap).sum(),
     (err_rep & swap & on_swapped_rep).sum((1, 2)) / (err_rep & swap).sum((1, 2)), 'rate'),
]
del on_swapped_rep

rows = []
for name, o, r, kind in STATS:
    o_str = summ(o, kind) if np.ndim(o) else num(o, kind)
    rows.append((name, o_str, summ(r, kind), fmt_p(np.mean(r >= o))))

out('## Table: test statistics')
out()
out('Observed value (posterior median [95% CI] where it depends on the draw), '
    'replicated value (median [95% CI] over draws), posterior predictive p.')
out()
out('| statistic | observed | replicated | p |')
out('|---|---|---|---|')
for row in rows:
    out('| ' + ' | '.join(row) + ' |')
out()
R.latex([r'\begin{table}[t]',
         r'  \centering\small',
         r'  \begin{tabular}{llll}',
         r'    \toprule',
         r'    Statistic & Observed & Replicated & $p$ \\',
         r'    \midrule']
        + [f'    {n} & {o} & {r} & {p_tex(p)} \\\\' for n, o, r, p in rows]
        + [r'    \bottomrule',
           r'  \end{tabular}',
           r'  \caption{Posterior predictive checks, pooled over all participants. '
           r'Replicated: posterior median and 95\% interval over replicate data sets, one '
           r'per joint posterior draw of latent trajectories and parameters. '
           r'$p = P(T(y^{\text{rep}}) \geq T(y^{\text{obs}}))$; values near 1 mean the model '
           r'predicts more of the statistic than observed. The log-likelihood is compared '
           r'draw by draw, so its observed value also has an interval.}',
           r'  \label{tab:ppc}',
           r'\end{table}'])


# ---------------------------------------------------------------- text ----

p_ll = (ppc['ll_rep'] >= ppc['ll_obs']).mean(0)               # (P,) per participant
nb = T // BLOCK
onehot = (cond[:, None] == np.arange(W)[None, :]).astype(float)        # (P, W)
blk_obs = err_obs.reshape(P, nb, BLOCK).sum(-1).T @ onehot             # (nb, W)
blk_rep = np.einsum('spb,pw->sbw', err_rep.reshape(S, P, nb, BLOCK).sum(-1), onehot)
p_blk = (blk_rep >= blk_obs[None]).mean(0)                              # (nb, W)

out('## Numbers in the text')
out()
out(f'- Per-participant log-likelihood p-values: median {np.median(p_ll):.2f}; '
    f'{(p_ll < 0.05).sum()} of {P} below 0.05, {(p_ll > 0.95).sum()} above 0.95.')
out(f'- Condition x {BLOCK}-trial block cells with p > 0.975 (fewer observed errors than '
    f'97.5% of replicates): {(p_blk > 0.975).sum()} of {p_blk.size}; with p < 0.025: '
    f'{(p_blk < 0.025).sum()}.')
out(f'- Error rate: observed {num(err_obs.mean(), "rate")}, replicated '
    f'{summ(err_rep.mean(axis=(1, 2)), "rate")}.')
out()


# ---------------------------------------------------------------- figure ----
# (a) error rate per block, all participants; (b) error rate by tied candidates;
# (c) observed - replicated error rate per word order, on all trials and on
#     the tied-candidate classes where the misfit sits

nS = AN.unpack_choices(ppc['n_surv_packed'], T).astype(np.int8) + 1    # (S, P, T)
tie_obs, tie_rep, tie_share = {}, {}, {}                                # k -> (S,) each
diff_w = {}                                                             # class -> (S, W)
n_w = onehot.sum(0)
diff_w['all'] = (err_obs.sum(1) @ onehot - err_rep.sum(2) @ onehot) / (n_w * T)
for k in TIES:
    m = nS == k
    n_tr = m.sum((1, 2))
    tie_obs[k] = (err_obs[None] & m).sum((1, 2)) / n_tr
    tie_rep[k] = (err_rep & m).sum((1, 2)) / n_tr
    tie_share[k] = n_tr / (P * T)
    n_tr_w = m.sum(2) @ onehot                                          # (S, W)
    diff_w[k] = ((err_obs[None] & m).sum(2) @ onehot
                 - (err_rep & m).sum(2) @ onehot) / np.maximum(n_tr_w, 1)
del nS

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5),
                         gridspec_kw={'width_ratios': [2.4, 1.6, 2.6]})
ax = axes[0]
xb = np.arange(nb) * BLOCK + BLOCK / 2
r_blk = blk_rep.sum(-1) / (P * BLOCK)                                   # (S, nb)
lo, md_, hi = np.percentile(r_blk, [2.5, 50, 97.5], axis=0)
ax.fill_between(xb, lo, hi, color='#888', alpha=0.3, lw=0, label='replicated (95%)')
ax.plot(xb, md_, color='#555', lw=1.2)
ax.plot(xb, blk_obs.sum(-1) / (P * BLOCK), 'ko', ms=3.5, label='observed')
ax.set_xlabel('trial', fontsize=8)
ax.set_ylabel('error rate', fontsize=8)
ax.set_title(f'(a) errors per {BLOCK}-trial block', fontsize=9)
ax.set_xlim(0, T)
ax.legend(fontsize=7, frameon=False)
ax.tick_params(labelsize=7)

ax = axes[1]
for j, k in enumerate(TIES):
    for off, d, style in ((-0.12, tie_obs[k], 'ko'), (0.12, tie_rep[k], 's')):
        lo, md_, hi = np.percentile(d, [2.5, 50, 97.5])
        kw = dict(color='k') if style == 'ko' else dict(color='#555', mfc='white')
        ax.errorbar(j + off, md_, yerr=[[md_ - lo], [hi - md_]], fmt=style[-1], ms=4,
                    capsize=0, lw=1, **kw)
# reference: error rate when guessing among the tied pictures, one of which
# is correct. The model sits above it on the 2-tied trials because of lapses
# and because ~4% of those states exclude the correct picture.
for j, k in enumerate(TIES):
    ax.plot([j - 0.3, j + 0.3], [1 - 1 / k] * 2, color='#999', ls='--', lw=0.8)
ax.set_xticks(range(len(TIES)))
ax.set_xticklabels([str(k) for k in TIES])
ax.set_xlabel('candidates tied in the sampled state', fontsize=8)
ax.set_title('(b) errors by tied candidates', fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(labelsize=7)
ax.plot([], [], 'ko', ms=4, label='observed')
ax.plot([], [], 's', color='#555', mfc='white', ms=4, label='replicated')
ax.plot([], [], color='#999', ls='--', lw=0.8, label='guessing among tied,\none correct')
ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.5)

ax = axes[2]
DIFF_CLASSES = [('all', 'all trials', 'o', 'k', 'k'), (2, '2 tied', 's', '#555', 'white'),
                (4, '4 tied', '^', '#555', '#bbb')]
for j, (key, lab, mk, ec, fc) in enumerate(DIFF_CLASSES):
    lo, md_, hi = np.percentile(diff_w[key], [2.5, 50, 97.5], axis=0)
    x = np.arange(W) + (j - 1) * 0.22
    ax.errorbar(x, md_, yerr=[md_ - lo, hi - md_], fmt=mk, ms=3.5, mec=ec, mfc=fc,
                ecolor=ec, capsize=0, lw=0.9, label=lab)
ax.axhline(0, color='#999', lw=0.7)
ax.set_xticks(range(W))
ax.set_xticklabels(labels)
ax.set_ylabel('observed $-$ replicated', fontsize=8)
ax.set_title('(c) error-rate gap by word order', fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc='lower right', ncol=1, handletextpad=0.2)
plt.tight_layout()
IO.savefig(fig, 'ppc.pdf')
IO.savefig(fig, 'ppc.png', dpi=200)

out('## Figure')
out()
out('`figures/ppc.pdf`: (a) error rate per block, observed (points) vs replicated '
    '(median and 95% band); (b) error rate by the number of candidates the sampled '
    'state leaves tied, observed vs replicated (median and 95% interval over draws; '
    'the observed rate varies over draws because the tie count does; dashed: error '
    'rate when guessing among the tied pictures, one of which is correct); (c) observed minus replicated error rate per word '
    'order, on all trials and on trials with 2 or 4 candidates tied.')
out()
out('Error-rate gap (observed - replicated) per word order, posterior median [95% CI]:')
out()
out('| order | all trials | 2 tied | 4 tied |')
out('|---|---|---|---|')
for w in range(W):
    out(f'| {labels[w]} | ' + ' | '.join(summ(diff_w[key][:, w], 'diff')
                                        for key in ('all', 2, 4)) + ' |')
out()
R.latex([r'\begin{figure}[t]',
         r'  \centering',
         r'  \includegraphics[width=\textwidth]{ppc.pdf}',
         r'  \caption{Posterior predictive checks. (a) Error rate per ' + str(BLOCK)
         + r'-trial block, pooled over participants: observed (points) and replicated '
         r'(line: median; band: 95\% interval). (b) Error rate by the number of candidate '
         r'pictures that the sampled latent state leaves equally compatible with the '
         r'utterance (4: no information), observed and replicated; intervals are over '
         r'posterior draws; dashed: error rate when guessing among the tied pictures, one '
         r'of which is correct. (c) Observed minus '
         r'replicated error rate per word-order condition, on all trials and on trials '
         r'with 2 or 4 candidates tied (posterior median and 95\% interval).}',
         r'  \label{fig:ppc}',
         r'\end{figure}'])


# ---------------------------------------------------------------- table: tied candidates ----

excess = {k: (tie_rep[k] - tie_obs[k]) * tie_share[k] * P * T for k in TIES}   # (S,) each
total_excess = sum(excess.values())
out('## Table: errors by number of tied candidates')
out()
out('| tied | share of trials | observed error rate | replicated error rate | '
    'excess replicated errors | share of total excess |')
out('|---|---|---|---|---|---|')
tie_rows = []
for k in TIES:
    cells = [str(k), num(np.median(tie_share[k]), 'rate'), summ(tie_obs[k], 'rate'),
             summ(tie_rep[k], 'rate'), f'{np.median(excess[k]):.0f}',
             num(np.median(excess[k] / total_excess), 'rate')]
    tie_rows.append(cells)
    out('| ' + ' | '.join(cells) + ' |')
out()
R.latex([r'\begin{table}[t]',
         r'  \centering\small',
         r'  \begin{tabular}{llllll}',
         r'    \toprule',
         r'    Tied & Share of trials & Observed errors & Replicated errors & '
         r'Excess errors & Share of excess \\',
         r'    \midrule']
        + ['    ' + ' & '.join(c) + r' \\' for c in tie_rows]
        + [r'    \bottomrule',
           r'  \end{tabular}',
           r'  \caption{Error rates by the number of candidate pictures the sampled latent '
           r'state leaves tied (posterior median and 95\% interval). Excess errors: replicated '
           r'minus observed errors, summed over all participants (posterior median).}',
           r'  \label{tab:ppc-ties}',
           r'\end{table}'])

R.save(IO.RESULTS_DIR / '05_ppc.md')
