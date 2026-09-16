"""
Per-participant PDF, one page per participant (sorted by accuracy):
smoothed posterior over word order, 7 stacked per-word belief lineplots
(posterior probability of each meaning per trial; line colour = meaning),
swap-trial dots, posterior-predictive P(wrong) with observed errors marked,
and the correctness strip. Title includes the PPC p-values.

Reads:  results/per_participant_marginals.pkl (+ per_participant_ppc.pkl if present)
Writes: figures/per_participant.pdf
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import data as RP
import helpers as HLP

order_labels = RP.order_labels_full
order_colors_full = RP.order_colors_full
MEANING_COLORS = RP.MEANING_COLORS
meaning_abbr = RP.meaning_abbr
word_names = RP.word_names
inv_true = RP.inv_true
true_order = RP.true_order
acc = RP.accuracy
choices = RP.choices
T = RP.T


def plot_page(pdf, p, res, swap_mask_p, ppc=None):
    bm = res['belief_marg']         # (T, 7, 8)  meanings 0..6 + UNK
    om = res['order_marg']          # (T, 7)
    params = res['params']          # natural (lam, eps_o, kappa, kappa_s, p_commit_o)
    lam, p_commit = float(params[0]), float(params[4])
    tm = inv_true[p]
    to = true_order[p]
    tail = om[-30:].mean(0)
    comm = tail[:6]
    final_o = int(comm.argmax()) if comm.max() > tail[6] else 6
    o_match = 'matches' if final_o == to else ('WRONG' if final_o < 6 else 'UNK')
    final_label = order_labels[final_o]

    CH = choices[:, p]
    correct = (CH == 0).astype(int)
    wrong_trials = np.where(correct == 0)[0]
    swap_trials = np.where(swap_mask_p)[0]
    swap_correct = correct[swap_mask_p].astype(bool)
    n_swap = swap_mask_p.sum()
    swap_acc = swap_correct.mean() if n_swap else float('nan')
    nonswap_acc = correct[~swap_mask_p].mean() if (~swap_mask_p).any() else float('nan')

    cmap_corr = ListedColormap(['#d73027', '#1a9850'])

    fig = plt.figure(figsize=(13, 8.4))
    # outer gridspec: order trace | word lineplots | swap dots | pred-wrong | correctness
    gs_outer = fig.add_gridspec(
        5, 1, height_ratios=[1.4, 2.0, 0.16, 0.55, 0.22], hspace=0.45)

    # --- top: per-trial P(order) trace ---
    ax_o = fig.add_subplot(gs_outer[0])
    for k in range(7):
        lw = 2.0 if k == to else (1.6 if k == 6 else 1.2)
        ls = ':' if k == 6 else '-'
        label = order_labels[k] + (' (true)' if k == to else '')
        ax_o.plot(np.arange(T), om[:, k], color=order_colors_full[k],
                  lw=lw, ls=ls, label=label)
    ax_o.axhline(0.5, color='#888', lw=0.5, ls='--')
    ax_o.set_ylim(0, 1.05)
    ax_o.set_ylabel('P(order|data)', fontsize=8)
    ax_o.set_title('Smoothed posterior over word order (incl UNK)', fontsize=9)
    ax_o.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),
                fontsize=7, frameon=False, handlelength=1.2)
    ax_o.set_xticks([])
    ax_o.set_xlim(-0.5, T - 0.5)

    # --- middle: 7 stacked lineplots, one per word; 7 lines each (one per
    # candidate meaning) showing P(belief[s, t] == meaning m) over trials.
    gs_words = gs_outer[1].subgridspec(7, 1, hspace=0.0)
    word_axes = []
    for s in range(7):
        ax_w = fig.add_subplot(gs_words[s], sharex=ax_o)
        for m in range(7):
            lw = 1.6 if m == tm[s] else 0.9
            alpha = 0.95 if m == tm[s] else 0.75
            ax_w.plot(np.arange(T), bm[:, s, m], color=MEANING_COLORS[m],
                      lw=lw, alpha=alpha)
        ax_w.set_ylim(0, 1.05)
        ax_w.set_yticks([])
        ax_w.set_xticks([])
        ax_w.set_xlim(-0.5, T - 0.5)
        ax_w.set_ylabel(f'{word_names[s]} → {meaning_abbr[tm[s]]}', rotation=0,
                        ha='right', va='center', fontsize=8,
                        color=MEANING_COLORS[tm[s]], fontweight='bold')
        for side in ('top', 'right'):
            ax_w.spines[side].set_visible(False)
        if s < 6:
            ax_w.spines['bottom'].set_visible(False)
        word_axes.append(ax_w)
    word_axes[0].set_title(
        'Per-word per-trial posterior P(belief[word] = meaning) — '
        '7 lines per row, one per meaning (colour = meaning); '
        'thicker line = the true meaning', fontsize=9)
    word_axes[0].set_yticks([1.0]); word_axes[0].set_yticklabels(['1'], fontsize=7)
    word_axes[-1].set_yticks([0.0]); word_axes[-1].set_yticklabels(['0'], fontsize=7)

    # --- swap-trial dots ---
    ax_s = fig.add_subplot(gs_outer[2], sharex=ax_o)
    if n_swap:
        col = np.where(swap_correct, '#1a9850', '#d73027')
        ax_s.scatter(swap_trials, np.zeros(n_swap), c=col, s=20,
                     edgecolor='k', linewidth=0.4, zorder=3)
    ax_s.set_ylim(-1, 1)
    ax_s.set_yticks([]); ax_s.set_xticks([])
    ax_s.set_xlim(-0.5, T - 0.5)
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    ax_s.set_ylabel('swap\ntrials', rotation=0, ha='right', va='center', fontsize=7)

    # --- predicted P(wrong) per trial (posterior predictive) ---
    ax_pw = fig.add_subplot(gs_outer[3], sharex=ax_o)
    if ppc is not None:
        pred_pw = ppc['pred_p_wrong']                    # (T,)
        ax_pw.plot(np.arange(T), pred_pw, color='#d73027', lw=1.0)
        if len(wrong_trials):                            # observed errors
            ax_pw.scatter(wrong_trials, pred_pw[wrong_trials], c='#d73027',
                          s=10, zorder=3, edgecolor='k', linewidth=0.3)
        ax_pw.axhline(0.75, color='#888', lw=0.4, ls=':')
    ax_pw.set_ylim(0, 1.02)
    ax_pw.set_yticks([0.0, 0.75])
    ax_pw.set_yticklabels(['0', '0.75'], fontsize=7)
    ax_pw.set_xticks([])
    ax_pw.set_xlim(-0.5, T - 0.5)
    ax_pw.set_ylabel('P(wrong)\npred + obs', rotation=0, ha='right',
                     va='center', fontsize=7)
    for side in ('top', 'right'):
        ax_pw.spines[side].set_visible(False)

    # --- correctness strip ---
    ax_c = fig.add_subplot(gs_outer[4], sharex=ax_o)
    ax_c.imshow(correct[None, :], cmap=cmap_corr, vmin=-0.5, vmax=1.5,
                aspect='auto', interpolation='nearest',
                extent=(-0.5, T - 0.5, 0.5, -0.5))
    ax_c.set_yticks([])
    ax_c.set_xlim(-0.5, T - 0.5)
    ax_c.set_xlabel('trial')
    ax_c.set_ylabel('correct', rotation=0, ha='right', va='center', fontsize=8)

    ppc_str = ''
    if ppc is not None:
        ppc_str = (f'  |  PPC: p(tot err)={ppc["p_total"]:.2f}, '
                   f'p(swap err)={ppc["p_swap"]:.2f}')
    fig.suptitle(
        f'p{p}  |  true {order_labels[to]}  |  acc {acc[p]:.0%}  '
        f'(non-swap {nonswap_acc:.0%}, swap {swap_acc:.0%} on n={n_swap})  |  '
        f'modal posterior: {final_label} ({o_match})  |  '
        f'lambda={lam:.3f}, p_commit={p_commit:.3f}{ppc_str}',
        fontsize=10.5, y=0.995)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def render_pdf(out_path):
    marg = HLP.load_derived('per_participant_marginals.pkl')
    print(f'  loaded {len(marg)} marginals', flush=True)
    ppc_data = None
    if (HLP.RESULTS_DIR / 'per_participant_ppc.pkl').exists():
        ppc_data = HLP.load_derived('per_participant_ppc.pkl')
        print('  loaded PPC', flush=True)
    swap_mask = HLP.compute_swap_mask()
    order = np.argsort(-acc)
    t0 = time.time()
    with PdfPages(str(out_path)) as pdf:
        for k, p in enumerate(order):
            p = int(p)
            ppc_p = None
            if ppc_data is not None:
                ppc_p = {'pred_p_wrong': ppc_data['pred_p_wrong'][p],
                         'p_total': float(ppc_data['p_total'][p]),
                         'p_swap': float(ppc_data['p_swap'][p])}
            plot_page(pdf, p, marg[p], swap_mask[:, p], ppc=ppc_p)
            if (k + 1) % 50 == 0:
                print(f'    {k + 1}/{len(order)}  ({time.time() - t0:.0f}s)', flush=True)
    print(f'  saved {out_path}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    HLP.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HLP.FIGURES_DIR / 'per_participant.pdf'
    print(f'Rendering {out_path.name}...', flush=True)
    render_pdf(out_path)
