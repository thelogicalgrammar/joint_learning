"""
Prior-vs-posterior predictive over per-participant parameters on the
natural scale, with practical interpretations annotated per panel.

For each of the 4 fitted parameters:
  - draw N new-participant values from the full hierarchy under the prior
    (tau^2 ~ IG, sigma^2 ~ IG, m ~ N(mu0, sd0^2), mu_w ~ N(m, tau^2),
    tpar ~ N(mu_w, sigma^2)) on the transformed scale, map to natural scale.
  - draw N new-participant values under the posterior, integrating over
    the hyperposterior samples (mu[w, k], sigma_k) with the condition w
    drawn uniformly.
  - overlay densities, mark medians, annotate with a directly-interpretable
    quantity per param:
        lambda     -> expected # lapses in 200 trials
        kappa      -> per-trial probability of any single-word belief move
        eps_o      -> expected # order switches in 200 trials post-commit
        gamma_o    -> trial by which 50% of UNK-state ppts have committed

kappa_s is fixed (model.KAPPA_S_FIXED), so it has no panel.

The prior config is read from the fit pickle (saved by fit_hierarchical.py).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, invgamma
import model as NB
import data as RP
import helpers as HLP
import config as CFG

T = RP.T

# Display settings per fitted coordinate, keyed by its config.PARAM_NAMES
# entry (the transformed -> natural mapping comes from model.tpar_to_nat_np).
#   (axis label, x range on the natural scale, log x axis)
DISPLAY = {
    'logit_lam':        (r'$\lambda$  (lapse)', (0, 0.4), False),
    'log_kappa':        (r'$\kappa$  (single-move log-cost)', (0, 15), False),
    'logit_eps_o':      (r'$\varepsilon_o$  (order-switch rate per trial)', (1e-4, 0.1), True),
    'logit_p_commit_o': (r'$\gamma_o$  (commit rate per trial)', (1e-3, 0.5), True),
}
NAMES = [DISPLAY[n][0] for n in CFG.PARAM_NAMES]
X_RANGES = [DISPLAY[n][1] for n in CFG.PARAM_NAMES]
LOG_X = [DISPLAY[n][2] for n in CFG.PARAM_NAMES]


# typical move counts for translating kappa to per-trial move probability
N_SINGLE_TYP, N_SWAP_TYP = 15.0, 5.0


def kappa_to_p_move(kappa, kappa_s=NB.KAPPA_S_FIXED):
    z = 1.0 + N_SINGLE_TYP * np.exp(-kappa) + N_SWAP_TYP * np.exp(-kappa_s)
    return 1.0 - 1.0 / z


def translate(med, k):
    """Translate a natural-scale median into a practical sentence."""
    name = CFG.PARAM_NAMES[k]
    if name == 'logit_lam':
        return f'~{med * T:.1f} lapses / {T} trials'
    if name == 'log_kappa':
        return f'P(belief move / trial) ~ {kappa_to_p_move(med):.2%}'
    if name == 'logit_eps_o':
        return f'~{med * T:.2f} order switches / {T} trials'
    if name == 'logit_p_commit_o':
        if med <= 0:
            return ''
        t50 = np.log(0.5) / np.log1p(-med)
        return f'50% have committed by trial {t50:.0f}'


# ---- load fit ----
fit = HLP.load_fit()
n_tpar = fit['n_tpar']
PRIOR_MU0, PRIOR_SD0 = fit['prior_mu0'], fit['prior_sd0']
A0, B0, A_TAU, B_TAU = fit['a0'], fit['b0'], fit['a_tau'], fit['b_tau']

mu_post = HLP.pool_hyper(fit, 'mu')                     # (S, W, K)
sig_post = HLP.pool_hyper(fit, 'sigma')                 # (S, K)
tpar_pmean = HLP.pool_chains(fit, 'tpar').mean(axis=0)  # (P, K)
S, W, _ = mu_post.shape

rng = np.random.default_rng(0)
N = 200000


def sample_prior_predictive(k, n):
    """n new-participant draws on tpar dim k from the full prior hierarchy."""
    tau2 = invgamma.rvs(A_TAU, scale=B_TAU, size=n, random_state=rng)
    sig2 = invgamma.rvs(A0, scale=B0, size=n, random_state=rng)
    m = rng.normal(PRIOR_MU0[k], PRIOR_SD0[k], size=n)
    mu_w = rng.normal(m, np.sqrt(tau2))
    return rng.normal(mu_w, np.sqrt(sig2))


def sample_posterior_predictive(k, n):
    """n new-participant draws on tpar dim k, integrating over the
    hyperposterior (a random sample s and a random condition w per draw)."""
    s = rng.integers(0, S, size=n)
    w = rng.integers(0, W, size=n)
    return rng.normal(mu_post[s, w, k], sig_post[s, k])


fig, axes = plt.subplots(1, n_tpar, figsize=(3.4 * n_tpar, 4.6))
for k in range(n_tpar):
    ax = axes[k]
    prior = NB.tpar_to_nat_np(sample_prior_predictive(k, N), k)
    post = NB.tpar_to_nat_np(sample_posterior_predictive(k, N), k)
    emp = NB.tpar_to_nat_np(tpar_pmean[:, k], k)

    lo, hi = X_RANGES[k]
    xx = np.geomspace(max(lo, 1e-8), hi, 400) if LOG_X[k] else np.linspace(lo, hi, 400)

    pr_in = prior[(prior >= lo) & (prior <= hi)]
    po_in = post[(post >= lo) & (post <= hi)]
    if len(pr_in) < 50 or len(po_in) < 50:
        ax.set_title(f'{NAMES[k]}\n(too few in range; check X_RANGES)')
        continue
    kd_pr = gaussian_kde(pr_in, bw_method=0.12)
    kd_po = gaussian_kde(po_in, bw_method=0.12)

    ymax = max(kd_pr(xx).max(), kd_po(xx).max())
    ax.fill_between(xx, kd_pr(xx), color='#3182bd', alpha=0.30, label='prior predictive')
    ax.plot(xx, kd_pr(xx), color='#3182bd', lw=1.4)
    ax.fill_between(xx, kd_po(xx), color='#e6550d', alpha=0.35, label='posterior predictive')
    ax.plot(xx, kd_po(xx), color='#e6550d', lw=1.4)

    m_pr, m_po = np.median(prior), np.median(post)
    ax.axvline(m_pr, color='#3182bd', ls='--', lw=1.0)
    ax.axvline(m_po, color='#e6550d', ls='--', lw=1.0)
    ax.scatter(emp, np.full(len(emp), -0.05 * ymax), marker='|', s=40,
               color='black', alpha=0.35,
               label=f'{len(emp)} ppts (post means)' if k == 0 else None)

    txt = (f'PRIOR  med = {m_pr:.4g}\n   {translate(m_pr, k)}\n\n'
           f'POST   med = {m_po:.4g}\n   {translate(m_po, k)}')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
            ha='right', va='top', family='monospace',
            bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.85,
                      boxstyle='round,pad=0.3'))
    if LOG_X[k]:
        ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_xlabel(NAMES[k], fontsize=10)
    ax.set_ylabel('density' if k == 0 else '')
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=-0.10 * ymax)
    if k == 0:
        ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.5),
                  ncol=3, frameon=False)

fig.suptitle('Hyperprior predictive: new-participant prior (blue) vs posterior '
             '(orange) on natural scale, conditions pooled; black ticks are '
             'per-ppt posterior means.', y=1.02, fontsize=11)
plt.tight_layout()
HLP.savefig(fig, 'hyperprior_predictive.png')
