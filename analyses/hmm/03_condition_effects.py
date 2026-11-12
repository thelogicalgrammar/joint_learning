"""
Results, section "Condition effects" (main text).

Everything this section of the paper reports is computed here, in the order
it appears in the paper:

  Figure   posterior per word order of (a) the commitment rate gamma,
           (b) the lexical change cost kappa, (c) the lapse rate lambda
  Figure   share of each condition's participants committed to the correct
           word order, by trial (from the sampled latent trajectories), with
           a table of the shares and of the trial by which half have committed
  Table    per word order: gamma, kappa, lambda
  Text     typological-group ratios (S-, V-, O-initial) for gamma, kappa and
           lambda; ranking of the six orders by gamma
  Table    all pairwise comparisons between word orders
  Tables   one per parameter: for every pair of orders, the share of
           posterior draws in which the row order's value exceeds the
           column order's

Natural-scale values are those of the *typical participant* of a condition:
the inverse link of the condition mean mu[j, k] (logit for gamma and lambda,
log for kappa), i.e. the median participant of condition j, not the mean
over participants. A group value (e.g. S-initial) is the average of the two
conditions' typical-participant values. Comparisons are ratios of these
values, with the 95% central posterior interval and P(ratio > 1).

gamma is the rate of leaving the UNK order state. In the model the order
committed to is uniform over the six; which order a participant actually
commits to is decided by the data (through the choices). So gamma is not the
rate of committing to the *correct* order, and no waiting time to correct
commitment follows from it. Commitment to the correct order is measured on
the sampled latent trajectories instead (second figure). Ratios of gamma are
the same as ratios of gamma / 6, so the comparisons are unaffected.

eps is not reported: the data do not identify it (parameter recovery,
appendix).

Reads:  results/hierarchical_fit.pkl
Writes: figures/condition_effects.pdf, figures/commitment_curves.pdf (+ .png)
        results/03_condition_effects.md   report: the numbers as Markdown
                                          tables and LaTeX to paste into the paper
"""
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm import model as NB
from jointlearn.hmm import config as CFG
from jointlearn.hmm.report import summ, fmt_p, p_tex, p_rel, Report

fit = IO.load_fit()
labels = list(fit['cond_labels'])                    # SVO, SOV, VSO, VOS, OSV, OVS
W = len(labels)
mu = IO.pool_hyper(fit, 'mu')                        # (S, W, K) transformed scale
S = mu.shape[0]


def typical(name):
    """(S, W) posterior of the typical participant's value of a parameter."""
    k = CFG.PARAM_NAMES.index(name)
    return NB.tpar_to_nat_np(mu[:, :, k], k=k)


gamma = typical('logit_p_commit_o')
kappa = typical('log_kappa')
lam = typical('logit_lam')
PARAMS = {'gamma': gamma, 'kappa': kappa, 'lambda': lam}


R = Report()
out = R.out


def prob(x):
    """P(x > 1) over posterior draws, as printed."""
    return fmt_p(np.mean(x > 1))


out('# Results: condition effects')
out()
out(f'Source: `03_condition_effects.py` on `{IO.FIT_NAME}` '
    f'({len(fit["chains"])} chains, {S} pooled posterior draws). '
    'Values are posterior medians [95% CI] for the typical participant of each '
    'condition (inverse link of the condition mean).')
out()


# ---------------------------------------------------------------- figure ----
# (a) gamma, (b) kappa, (c) lambda; one violin per order, plus the prior of a
# condition mean (m + tau * z under the hyperpriors) in grey for scale. Drawn
# on the link scale (logit / log), where the fit lives and the prior is
# Gaussian-like, with natural-scale tick labels.

def prior_typical(name, n=20000, seed=0):
    """Prior draws of a condition mean mu[w, k] (transformed scale)."""
    k = CFG.PARAM_NAMES.index(name)
    r = np.random.default_rng(seed)
    m = r.normal(CFG.PRIOR_MU0[k], CFG.PRIOR_SD0[k], n)
    tau = np.sqrt(1.0 / r.gamma(CFG.A_TAU, 1.0 / CFG.B_TAU, n))
    return m + tau * r.normal(size=n)


colors = RP.order_colors
fig = plt.figure(figsize=(7.0, 2.0))
gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.065, right=0.955, bottom=0.12, top=0.88)
panels = [('logit_p_commit_o', gamma, r'(a) commitment rate $\gamma$', r'$\gamma$', True,
           [1e-6, 1e-4, 1e-2, 0.5, 0.99]),
          ('log_kappa', kappa, r'(b) lexical change cost $\kappa$', r'$\kappa$', False,
           [0.01, 1, 100]),
          ('logit_lam', lam, r'(c) lapse rate $\lambda$', r'$\lambda$', False,
           [1e-3, 0.01, 0.1, 0.5, 0.9])]
for g, (name, x, title, ylabel, log, ticks) in zip(gs, panels):
    sub = g.subgridspec(1, 2, width_ratios=[4.2, 1], wspace=0.05)
    ax, ax_p = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    # right: prior vs posterior (all conditions pooled) on the link scale
    k = CFG.PARAM_NAMES.index(name)
    link = np.log if NB.TPAR_SPEC[k][1] == 'log' else (lambda v: np.log(v / (1 - v)))
    prior = prior_typical(name)
    AN.violins(ax_p, [prior, mu[:, :, k].ravel()], ['prior', 'post'], ['0.6', '0.25'],
               width=0.7)
    ax_p.set_ylim(np.percentile(prior, [0.5, 99.5]))
    ax_p.set_yticks(link(np.asarray(ticks, float)))
    ax_p.set_yticklabels([f'{t:g}' for t in ticks])
    ax_p.yaxis.tick_right()
    ax_p.tick_params(labelsize=5.5)
    ax_p.set_xticks([0, 1], ['prior', 'post'], fontsize=5, rotation=90)
    ax_p.tick_params(axis='x', pad=1, length=0)
    # left: posterior per word order, zoomed
    AN.violins(ax, [x[:, w] for w in range(W)], labels, colors)
    if log:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=9, x=0.62)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.tick_params(axis='x', labelsize=5.5, pad=2)
IO.savefig(fig, 'condition_effects.pdf')
IO.savefig(fig, 'condition_effects.png', dpi=200)

out('## Figure')
out()
out('`figures/condition_effects.pdf`: posterior per word order of (a) gamma '
    '(log scale), (b) kappa, (c) lambda; right of each, the prior of a condition mean vs '
    'the posterior pooled over conditions (logit / log scale).')
out()
out('```latex')
out(r'\begin{figure}[t]')
out(r'  \centering')
out(r'  \includegraphics[width=\textwidth]{condition_effects.pdf}')
out(r'  \caption{Posterior distributions for the typical participant of each word-order '
    r'condition. (a) Per-trial rate $\gamma$ of committing to a word order (log scale). '
    r'(b) Cost of a lexical change $\kappa$. (c) Lapse rate $\lambda$. Narrow panels: '
    r'prior for a condition mean (light grey) and posterior pooled over conditions (dark '
    r'grey), on the logit ($\gamma$, $\lambda$) or log ($\kappa$) scale. Bars: median '
    r'and 95\% interval.}')
out(r'  \label{fig:condition-effects}')
out(r'\end{figure}')
out('```')
out()


# ---------------------------------------------------------------- figure: commitment curves ----
# Share of a condition's participants whose sampled word-order state is the
# correct order on each trial, per posterior draw; curve = posterior mean
# (expected share), band = 95% interval. Unlike the typical-participant
# values this uses every participant's sampled trajectory. Half-time: the
# first trial on which at least half of the condition is committed to the
# correct order (per draw; > T if never within the session).

O = IO.pool_chains(fit, 'O')                         # (S, P, T) int8
cond = np.asarray(RP.true_order)
committed = np.stack([(O[:, cond == w] != NB.UNK_O).mean(axis=1)
                      for w in range(W)], axis=1)    # (S, W, T)
true_ord = np.stack([(O[:, cond == w] == w).mean(axis=1)
                     for w in range(W)], axis=1)     # (S, W, T), committed to the true order
del O

trial = np.arange(1, RP.T + 1)
fig, ax = plt.subplots(figsize=(5.0, 2.8))
for w in range(W):
    lo, hi = np.percentile(true_ord[:, w], [2.5, 97.5], axis=0)
    ax.fill_between(trial, lo, hi, color=colors[w], alpha=0.2, lw=0)
    ax.plot(trial, true_ord[:, w].mean(axis=0), color=colors[w], lw=1.4, label=labels[w])
ax.set_xlim(1, RP.T)
ax.set_ylim(0, 1)
ax.set_xlabel('trial', fontsize=8)
ax.axhline(0.5, color='#999', ls=':', lw=0.7)
ax.set_ylabel('share committed to correct order', fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
plt.tight_layout()
IO.savefig(fig, 'commitment_curves.pdf')
IO.savefig(fig, 'commitment_curves.png', dpi=200)

out('## Figure: commitment curves')
out()
out('`figures/commitment_curves.pdf`: share of each condition\'s participants '
    'committed to the correct word order on each trial (posterior mean, 95% band; '
    'dotted: half). Table: posterior median [95% CI]; half-time = first trial on '
    'which at least half of the condition is committed to the correct order.')
out()


def half_time(share):
    """(S, T) share per draw -> (S,) first trial with share >= 0.5 (inf if never)."""
    hit = share >= 0.5
    return np.where(hit.any(-1), hit.argmax(-1) + 1, np.inf)


def summ_time(t):
    """summ() for half-times, '> T' where the session ends first."""
    q = np.percentile(t, [50, 2.5, 97.5], method='nearest')
    f = [f'> {RP.T}' if np.isinf(v) else f'{v:.0f}' for v in q]
    return f'{f[0]} [{f[1]}, {f[2]}]'


ct_rows = []
for w in range(W):
    ht = half_time(true_ord[:, w])
    ct_rows.append([labels[w], summ(true_ord[:, w, 49], 'rate'), summ(true_ord[:, w, -1], 'rate'),
                    summ(committed[:, w, -1], 'rate'), summ_time(ht), fmt_p(np.mean(np.isfinite(ht)))])
out('| order | correct order, trial 50 | correct order, trial 200 | any order, trial 200 | '
    'half-time (trial) | P(half reached by 200) |')
out('|---|---|---|---|---|---|')
for row in ct_rows:
    out('| ' + ' | '.join(row) + ' |')
out()
R.latex([r'\begin{table}[t]',
         r'  \centering\small',
         r'  \begin{tabular}{llllll}',
         r'    \toprule',
         r'    & \multicolumn{2}{c}{Correct order} & Any order & & \\',
         r'    \cmidrule(lr){2-3}\cmidrule(lr){4-4}',
         r'    Order & Trial 50 & Trial 200 & Trial 200 & Half-time & $P(\text{half by 200})$ \\',
         r'    \midrule']
        + ['    ' + ' & '.join(r[:5] + [p_tex(r[5])]).replace('> 200', '$>$ 200') + r' \\'
           for r in ct_rows]
        + [r'    \bottomrule',
           r'  \end{tabular}',
           r'  \caption{Share of participants committed to the correct word order (and to any '
           r'order) at trials 50 and 200, and the first trial by which half of a condition has '
           r'committed to the correct order; posterior median and 95\% interval over draws of '
           r'the latent word-order trajectories.}',
           r'  \label{tab:commitment}',
           r'\end{table}'])
out('```latex')
out(r'\begin{figure}[t]')
out(r'  \centering')
out(r'  \includegraphics[width=0.65\textwidth]{commitment_curves.pdf}')
out(r'  \caption{Share of participants in each word-order condition who have committed to '
    r'the correct word order, by trial. Lines: posterior mean; bands: 95\% interval over '
    r'posterior draws of the latent word-order trajectories; dotted: half of the condition.}')
out(r'  \label{fig:commitment-curves}')
out(r'\end{figure}')
out('```')
out()


# ---------------------------------------------------------------- table: per word order ----

cols = [('gamma', gamma, 'gamma'), ('kappa', kappa, 'kappa'), ('lambda', lam, 'lambda')]
out('## Table: posterior per word order')
out()
out('| order | ' + ' | '.join(c for c, _, _ in cols) + ' |')
out('|---|' + '---|' * len(cols))
for w in range(W):
    out(f'| {labels[w]} | ' + ' | '.join(summ(x[:, w], k) for _, x, k in cols) + ' |')
out()
out('```latex')
out(r'\begin{table}[t]')
out(r'  \centering\small')
out(r'  \begin{tabular}{llll}')
out(r'    \toprule')
out(r'    Order & $\gamma$ & $\kappa$ & $\lambda$ \\')
out(r'    \midrule')
for w in range(W):
    out(f'    {labels[w]} & ' + ' & '.join(summ(x[:, w], k) for _, x, k in cols) + r' \\')
out(r'    \bottomrule')
out(r'  \end{tabular}')
out(r'  \caption{Posterior median and 95\% interval for the typical participant of each '
    r'word-order condition. $\gamma$: per-trial rate of committing to a word order; '
    r'$\kappa$: cost of a lexical change; $\lambda$: lapse rate.}')
out(r'  \label{tab:condition-posteriors}')
out(r'\end{table}')
out('```')
out()


# ---------------------------------------------------------------- text ----
# typological groups: average of the two conditions' typical-participant values

GROUPS = [('S-initial', 'V-initial'), ('V-initial', 'O-initial'), ('S-initial', 'O-initial')]
out('## Numbers in the text')
out()
out('Typological groups (S-initial = SVO, SOV; V-initial = VSO, VOS; '
    'O-initial = OSV, OVS): ratio of the group values, 95% CI, P(ratio > 1).')
out()
out('| parameter | comparison | ratio [95% CI] | P(ratio > 1) |')
out('|---|---|---|---|')
snippets = []
for name, x in PARAMS.items():
    g = {grp: x[:, idx].mean(axis=1) for grp, idx in RP.TYPOLOGY.items()}
    for a, b in GROUPS:
        r = g[a] / g[b]
        out(f'| {name} | {a} / {b} | {summ(r, "ratio")} | {prob(r)} |')
        snippets.append((f'{name}, {a} / {b}',
                         f'{summ(r, "ratio")}, $P(\\text{{ratio}} > 1) {p_rel(prob(r))}$'))
out()

order = np.argsort(-np.median(gamma, axis=0))
adj = [f'P({labels[a]} > {labels[b]}) = {prob(gamma[:, a] / gamma[:, b])}'
       for a, b in zip(order[:-1], order[1:])]
out('Ranking of the orders by the posterior median of gamma (fastest commitment first): '
    + ' > '.join(labels[w] for w in order) + '.')
out('Posterior probability of each adjacent step: ' + '; '.join(adj) + '.')
out()
out('LaTeX for the in-text numbers (ratio [95% CI], posterior probability):')
out()
out('```latex')
for lab, s in snippets:
    out(f'% {lab}')
    out(s)
out('```')
out()


# ---------------------------------------------------------------- table: pairwise ----

pairs = list(combinations(range(W), 2))
out('## Table: all pairwise comparisons')
out()
out('Ratio of the typical-participant values (row order / column order as '
    'listed), 95% CI, P(ratio > 1). For gamma, a ratio > 1 means the first '
    'order commits faster.')
out()
out('| comparison | ' + ' | '.join(f'{n} ratio | P' for n in PARAMS) + ' |')
out('|---|' + '---|---|' * len(PARAMS))
rows_tex = []
for a, b in pairs:
    cells_md, cells_tex = [], []
    for name, x in PARAMS.items():
        r = x[:, a] / x[:, b]
        cells_md += [summ(r, 'ratio'), prob(r)]
        cells_tex += [summ(r, 'ratio'), p_tex(prob(r))]
    out(f'| {labels[a]} / {labels[b]} | ' + ' | '.join(cells_md) + ' |')
    rows_tex.append(f'    {labels[a]} / {labels[b]} & ' + ' & '.join(cells_tex) + r' \\')
out()
out('```latex')
out(r'\begin{table}[t]')
out(r'  \centering\footnotesize')
out(r'  \begin{tabular}{lllllll}')
out(r'    \toprule')
out(r'    & \multicolumn{2}{c}{$\gamma$} & \multicolumn{2}{c}{$\kappa$} & \multicolumn{2}{c}{$\lambda$} \\')
out(r'    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
out(r'    Comparison & Ratio & $P$ & Ratio & $P$ & Ratio & $P$ \\')
out(r'    \midrule')
for row in rows_tex:
    out(row)
out(r'    \bottomrule')
out(r'  \end{tabular}')
out(r'  \caption{Pairwise comparisons between word-order conditions: ratio of the '
    r'typical-participant values (posterior median and 95\% interval) and posterior '
    r'probability that the ratio exceeds 1.}')
out(r'  \label{tab:pairwise}')
out(r'\end{table}')
out('```')

# ---------------------------------------------------------------- tables: P(row > column) ----
# Share of posterior draws in which the row order's typical-participant value
# exceeds the column order's. The inverse link is monotone, so this equals the
# same probability for the condition means mu on the transformed scale.

SYMBOL = {'gamma': r'$\gamma$', 'kappa': r'$\kappa$', 'lambda': r'$\lambda$'}
out('## Tables: P(row order > column order), per parameter')
out()
out('Share of posterior draws in which the row order\'s value is higher than the '
    'column order\'s; each cell equals 1 minus its mirror image.')
out()
for name, x in PARAMS.items():
    cells = [['—' if a == b else fmt_p(np.mean(x[:, a] > x[:, b])) for b in range(W)]
             for a in range(W)]
    out(f'### {name}')
    out()
    out('| row > column | ' + ' | '.join(labels) + ' |')
    out('|---|' + '---|' * W)
    for a in range(W):
        out(f'| {labels[a]} | ' + ' | '.join(cells[a]) + ' |')
    out()
    R.latex([r'\begin{table}[t]',
             r'  \centering\small',
             r'  \begin{tabular}{l' + 'c' * W + '}',
             r'    \toprule',
             r'    Row $>$ column & ' + ' & '.join(labels) + r' \\',
             r'    \midrule']
            + [f'    {labels[a]} & ' + ' & '.join('---' if c == '—' else p_tex(c) for c in cells[a])
               + r' \\' for a in range(W)]
            + [r'    \bottomrule',
               r'  \end{tabular}',
               r'  \caption{Posterior probability that ' + SYMBOL[name] + r' of the typical '
               r'participant is higher in the row word order than in the column word order.}',
               r'  \label{tab:pgreater-' + name + '}',
               r'\end{table}'])

R.save(IO.RESULTS_DIR / '03_condition_effects.md')
