"""
Per-participant latent trajectories: the paper figure with four example
participants, and a supplementary PDF with every participant.

One panel per participant, all from the joint posterior:
  order    posterior probability per trial that the word-order state is the
           correct order, another (wrong) order, or UNK (stacked)
  words    one strip per word: posterior probability that its belief is the
           correct meaning, a wrong meaning, or UNK (stacked)
  P(error) the error probability implied by the inferred latent state (line:
           posterior mean over joint draws of P(error | B_t, O_t, lambda)), the
           observed errors (ticks), and the role-swap trials (triangles,
           filled when the participant erred on them). The latent states are
           inferred from all of the participant's choices, including trial t's,
           so the line is an in-sample fit, not a one-step-ahead prediction.

Paper figure: four participants chosen to show four patterns, labelled by
panel letter (participant numbers mean nothing to a reader; they are indices
into the analysed dataset and label the supplement's panels; the report lists
the numbers that motivated each choice):
  (a) p218  SVO, fast: commits to the correct order within a few trials
  (b) p212  SOV, gradual: words learned one by one, order committed around trial 77
  (c) p41   SOV, commits to a wrong order and holds several wrong meanings
  (d) p226  OSV, learns every word but never commits to an order, yet 92% accurate:
            its errors fall on the role-swap trials, where word order is needed

Reads:  results/per_participant_marginals.pkl, results/per_participant_ppc.pkl
Writes: figures/participants.pdf (+ .png)   paper figure
        figures/per_participant.pdf         supplement: all participants, 6 per page,
                                            by condition, then accuracy (high first)
        results/06_per_participant.md       report: why these four, and LaTeX
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from jointlearn.hmm import dataset as RP
from jointlearn.hmm import io as IO, analysis as AN
from jointlearn.hmm.report import Report

PAPER_PARTICIPANTS = [218, 212, 41, 226]             # panels (a)-(d)
LETTERS = 'abcdefgh'
C_OK, C_BAD, C_UNK = '#2b8cbe', '#f03b20', '#f0f0f0'          # correct / wrong / UNK
PANEL_W, PANEL_H = 3.4, 2.4                                    # inches per panel

marg = IO.load_derived('per_participant_marginals.pkl')
ppc = IO.load_derived('per_participant_ppc.pkl')
swap_mask = AN.compute_swap_mask()                             # (T, P)
T, P = RP.T, RP.P
trial = np.arange(1, T + 1)


def _strip(ax, ok, bad):
    """Stacked correct / wrong / UNK probabilities over trials, no axes."""
    ax.stackplot(trial, ok, bad, 1 - ok - bad, colors=[C_OK, C_BAD, C_UNK], lw=0)
    ax.set_xlim(1, T)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def panel(fig, spec, p, left=True, bottom=True, compact=False, name=None):
    """Draw participant p's panel into the gridspec cell `spec`. Row labels
    only if `left`, trial axis only if `bottom`; `compact` = thinner strips
    (paper figure); `name` replaces the participant number in the title."""
    bm, om = marg[p]['belief_marg'], marg[p]['order_marg']    # (T, 7, 8), (T, 7)
    to, tm = RP.true_order[p], RP.inv_true[p]
    ratios, gap = ([0.4, 1.3, 0.8], 0.1) if compact else ([0.5, 1.6, 0.9], 0.25)
    gs = spec.subgridspec(3, 1, height_ratios=ratios, hspace=0.12)

    def row_label(ax, text, size):
        if left:
            ax.set_ylabel(text, rotation=0, ha='right', va='center', fontsize=size)

    ax = fig.add_subplot(gs[0])
    _strip(ax, om[:, to], om[:, :6].sum(1) - om[:, to])
    row_label(ax, 'order', 6)
    ax.set_title(f'{name or f"p{p},"} {RP.order_labels[to]}, accuracy {RP.accuracy[p]:.0%}',
                 fontsize=6.5 if compact else 7, loc='left', pad=2)

    gw = gs[1].subgridspec(7, 1, hspace=gap)
    for s in range(7):
        ax = fig.add_subplot(gw[s])
        ok = bm[:, s, tm[s]]
        _strip(ax, ok, bm[:, s, :7].sum(1) - ok)
        row_label(ax, RP.word_names[s], 5.5)

    ax = fig.add_subplot(gs[2])
    err = RP.choices[:, p] != 0
    sw = swap_mask[:, p]
    ax.plot(trial, ppc['pred_p_wrong'][p], color='#555', lw=0.7)
    ax.vlines(trial[err], 1.04, 1.16, color=C_BAD, lw=0.6)
    for erred, face in ((False, 'white'), (True, C_BAD)):
        m = sw & (err == erred)
        ax.plot(trial[m], np.full(m.sum(), -0.12), '^', ms=2.4, mew=0.4,
                mfc=face, mec='#333', ls='none')
    ax.set_xlim(1, T)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 0.75])
    ax.tick_params(labelsize=5, length=2)
    if left:
        ax.set_ylabel('P(error)', fontsize=6)
    else:
        ax.set_yticklabels([])
    if bottom:
        ax.set_xlabel('trial', fontsize=6, labelpad=1)
    else:
        ax.set_xticklabels([])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def legend(fig, y=-0.02):
    fig.legend(handles=[Patch(color=C_OK, label='correct'), Patch(color=C_BAD, label='wrong'),
                        Patch(facecolor=C_UNK, edgecolor='#bbb', lw=0.5, label='UNK'),
                        Line2D([], [], color='#555', lw=0.7, label='implied P(error)'),
                        Line2D([], [], color=C_BAD, marker='|', ls='none', label='observed error'),
                        Line2D([], [], marker='^', mfc='white', mec='#333', ls='none', ms=3,
                               label='role-swap trial')],
               loc='lower center', ncol=6, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, y))


def grid(ps, ncol=2):
    """Supplement page: one labelled panel per participant, ncol per row."""
    nrow = int(np.ceil(len(ps) / ncol))
    fig = plt.figure(figsize=(PANEL_W * ncol, PANEL_H * nrow + 0.3))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.35, wspace=0.3)
    for i, p in enumerate(ps):
        panel(fig, gs[i // ncol, i % ncol], p)
    legend(fig)
    return fig


def paper_grid(ps, ncol, size, wspace, hspace):
    """Paper figure: compact panels, row labels on the left column and the
    trial axis on the bottom row only."""
    nrow = int(np.ceil(len(ps) / ncol))
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(nrow, ncol, wspace=wspace, hspace=hspace)
    for i, p in enumerate(ps):
        r, c = divmod(i, ncol)
        panel(fig, gs[r, c], p, left=c == 0, bottom=r == nrow - 1, compact=True,
              name=f'({LETTERS[i]})')
    legend(fig, y=-0.1)
    return fig


def summary(p):
    """The numbers behind the choice of an example participant."""
    om, to = marg[p]['order_marg'], RP.true_order[p]
    tail = om[-30:].mean(0)                                   # last 30 trials
    hit = om[:, to] >= 0.5
    half = f'{int(hit.argmax()) + 1}' if hit.any() else 'never'
    return (f'p{p}', RP.order_labels[to], f'{RP.accuracy[p]:.2f}', half,
            f'{tail[to]:.2f}', f'{tail[:6].sum() - tail[to]:.2f}', f'{tail[6]:.2f}')


# ---------------------------------------------------------------- paper figure ----

R = Report()
out = R.out
out('# Results: example participants')
out()
fig = paper_grid(PAPER_PARTICIPANTS, ncol=4, size=(7.0, 2.3), wspace=0.06, hspace=0.2)
IO.savefig(fig, 'participants.pdf')
IO.savefig(fig, 'participants.png', dpi=200)
plt.close(fig)

out('Why these four: accuracy, first trial with P(correct order) >= 0.5, and the '
    'mean posterior over the last 30 trials of the order state.')
out()
out('| panel | participant | condition | accuracy | correct order >= 0.5 from trial | '
    'correct order | wrong order | UNK |')
out('|---|---|---|---|---|---|---|---|')
for i, p in enumerate(PAPER_PARTICIPANTS):
    out(f'| ({LETTERS[i]}) | ' + ' | '.join(summary(p)) + ' |')
out()
R.latex([r'\begin{figure}[t]',
         r'  \centering',
         r'  \includegraphics[width=\textwidth]{participants.pdf}',
         r'  \caption{Latent trajectories of four participants. Top strip: posterior '
         r'probability that the word-order state is the correct order (blue), a wrong order '
         r"(red) or uncommitted (grey). Seven strips: the same for each word's "
         r"meaning. Bottom: the probability of an error implied by the inferred latent state "
         r"(line; posterior mean given all of the participant's choices, including that "
         r"trial's), observed errors (ticks) and role-swap trials, where word "
         r'order is needed to choose correctly (triangles; filled: error). '
         r'(a) Fast commitment to the correct order. (b) Gradual learning of words and '
         r'order. (c) Commitment to a wrong order, with several wrong word meanings. '
         r'(d) All words learned but no commitment to an order: errors fall on the '
         r'role-swap trials.}',
         r'  \label{fig:participants}',
         r'\end{figure}'])
R.save(IO.RESULTS_DIR / '06_per_participant.md')


# ---------------------------------------------------------------- supplement ----

PER_PAGE = 6
order = sorted(range(P), key=lambda p: (RP.true_order[p], -RP.accuracy[p]))
with PdfPages(str(IO.FIGURES_DIR / 'per_participant.pdf')) as pdf:
    for i in range(0, P, PER_PAGE):
        fig = grid(order[i:i + PER_PAGE])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
print(f'Saved: {IO.FIGURES_DIR / "per_participant.pdf"} ({int(np.ceil(P / PER_PAGE))} pages)')
