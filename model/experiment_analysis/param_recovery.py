"""
Parameter recovery for the hierarchical HMM fit.

Synthetic datasets are the real experiment (same 325 participants, stimuli
and word-order conditions) with the choices replaced by draws from the
model, so the fit script runs on them unchanged (data.py, env EA_SIM_DATA).
Three ways of choosing the ground truth (`--mode`):

  prior      condition means / sds fixed at the posterior means of the real
             fit (mu, sigma, m, tau); per-participant tpar ~ N(mu[w(p)],
             sigma^2); belief and order trajectories drawn forward from the
             transition model (simulate.simulate_latents); choices from the
             emission model. This is the model's own generative process.
             Beware: it is a measurement model, so prior-simulated lexicons
             are random walks that do not converge to the truth (accuracy
             against the true scene stays near chance), and a wrong lexicon
             usually contradicts all four candidate scenes, which makes the
             emission uniform: ~77% of prior-simulated trials carry no
             information about the state (vs ~19% in `posterior` mode).
             Recovery is therefore much weaker here (order trajectories and
             eps_o essentially unidentified, contrasts attenuated); treat it
             as a sampler sanity check, not as the recovery result.
  null       as `prior` but with all six condition means equal to the
             population mean m: a false-positive check for the condition
             contrasts (analyze_condition_effects.py).
  posterior  one joint posterior draw (B, O, tpar, hyper) of the real fit is
             the truth and only the choices are regenerated from it. The
             latent trajectories are then learning-like (lexicon converging
             to the true one, order committing to the true order), i.e. the
             regime of the real data; the truth is posterior-typical rather
             than prior-typical.

Usage (from this directory; the real results/hierarchical_fit.pkl is needed
by every mode for the ground truth):

  python param_recovery.py simulate --mode prior --n_rep 8
  python param_recovery.py fit      --mode prior --rep 0      # one dataset
  python param_recovery.py fit      --mode prior               # all datasets
  python param_recovery.py analyze  --mode prior

`fit` runs fit_hierarchical.py in a subprocess with EA_SIM_DATA and
EA_RESULTS_DIR pointing at results/param_recovery/<mode>/rep<k>/ (fit
pickle, checkpoints, fit.log). Its MCMC config comes from the usual env vars
with recovery defaults N_OUTER=1200 BURN=400 THIN=8 N_CHAINS=3 RESUME=1, so
an interrupted or re-submitted job continues from its checkpoints. On
Snellius, ../../server_jobs/slurm_param_recovery.sh fans `fit --rep k` out
as an array job.

`analyze` compares, per fitted coordinate, the posterior of the condition
means and of the per-participant tpar with the truth (coverage of the 95%
interval, bias, correlation), the typological p_commit contrasts of
analyze_condition_effects.py, the hyper sds, and the latent trajectories
(posterior probability of the true order / belief per trial, commit-time
recovery). Writes figures/param_recovery_<mode>.png.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np

if os.environ.get('EA_SIM_DATA'):
    raise SystemExit('param_recovery.py must run on the real data: unset EA_SIM_DATA')

import data as RP
import helpers as HLP
import model as NB
import config as CFG
import simulate as SIM

MODES = ('prior', 'null', 'posterior')
PR_DIR = HLP.RESULTS_DIR / 'param_recovery'
HERE = Path(__file__).resolve().parent
FIT_DEFAULTS = dict(N_OUTER='1200', BURN='400', THIN='8', N_CHAINS='3',
                    CKPT_EVERY='200', RESUME='1')
W, K, T, P = len(RP.order_labels), CFG.N_TPAR, RP.T, RP.P
I_COMMIT = CFG.PARAM_NAMES.index('logit_p_commit_o')
CONTRASTS = [('S-initial', 'V-initial'), ('V-initial', 'O-initial'),
             ('S-initial', 'O-initial')]


def rep_dir(mode, rep):
    return PR_DIR / mode / f'rep{rep:02d}'


def sim_path(mode, rep):
    return rep_dir(mode, rep) / 'sim_data.npz'


def reps_available(mode):
    return sorted(int(d.name[3:]) for d in (PR_DIR / mode).glob('rep[0-9][0-9]')
                  if (d / 'sim_data.npz').exists()) if (PR_DIR / mode).exists() else []


# ------------------------------------------------------------- simulate ----

def simulate(mode, n_rep, seed):
    fit = HLP.load_fit()
    mu_s, sig_s = HLP.pool_hyper(fit, 'mu'), HLP.pool_hyper(fit, 'sigma')
    m_s, tau_s = HLP.pool_hyper(fit, 'm'), HLP.pool_hyper(fit, 'tau')
    U = np.ascontiguousarray(RP.signals.transpose(1, 0, 2))            # (P, T, 3)
    C = np.ascontiguousarray(RP.cand_mean.transpose(1, 0, 2, 3))       # (P, T, 4, 3)
    WO = np.asarray(RP.word_orders)
    cond = RP.true_order
    if mode == 'posterior':
        tpar_s = HLP.pool_chains(fit, 'tpar')
        B_s, O_s = HLP.pool_chains(fit, 'B'), HLP.pool_chains(fit, 'O')
    for rep in range(n_rep):
        rng = np.random.default_rng([seed, MODES.index(mode), rep])
        t0 = time.time()
        if mode == 'posterior':
            s = int(rng.integers(tpar_s.shape[0]))
            B, O, tpar = B_s[s], O_s[s], tpar_s[s].astype(np.float64)
            truth = dict(mu=mu_s[s], sigma=sig_s[s], m=m_s[s], tau=tau_s[s], sample=s)
        else:
            mu = mu_s.mean(0) if mode == 'prior' else np.tile(m_s.mean(0), (W, 1))
            sigma = sig_s.mean(0)
            tpar = rng.normal(mu[cond], sigma)                          # (P, K)
            B = np.zeros((P, T, NB.N_W), np.int8)
            O = np.zeros((P, T), np.int8)
            for p in range(P):
                B[p], O[p] = SIM.simulate_latents(rng, tpar[p], U[p])
            truth = dict(mu=mu, sigma=sigma, m=m_s.mean(0), tau=tau_s.mean(0), sample=-1)
        lam = NB.tpar_to_nat_np(tpar[:, 0], 0)
        CH = SIM.simulate_choices(rng, B, O, lam, U, C, WO)             # (P, T)
        truth.update(B=B, O=O, tpar=tpar.astype(np.float32), cond=cond,
                     mode=mode, seed=seed, rep=rep)
        rep_dir(mode, rep).mkdir(parents=True, exist_ok=True)
        SIM.save_sim_dataset(sim_path(mode, rep), CH.T, truth)
        acc = (CH == 0).mean()
        n_comm = (O[:, -1] != NB.UNK_O).mean()
        print(f'[{mode} rep {rep}] wrote {sim_path(mode, rep).name}: mean accuracy '
              f'{acc:.3f}, {n_comm:.0%} ppts committed by trial {T}, '
              f'{(B[:, -1] != -1).mean():.2f} of lexicon slots committed at the end '
              f'({time.time() - t0:.0f}s)', flush=True)


# ------------------------------------------------------------------ fit ----

def fit(mode, reps, force):
    for rep in reps:
        d = rep_dir(mode, rep)
        out = d / HLP.FIT_NAME
        if out.exists() and not force:
            print(f'[{mode} rep {rep}] {out} exists — skipping (use --force to refit)')
            continue
        env = dict(os.environ)
        for k, v in FIT_DEFAULTS.items():
            env.setdefault(k, v)
        env.update(EA_SIM_DATA=str(sim_path(mode, rep)), EA_RESULTS_DIR=str(d),
                   EA_FIGURES_DIR=str(d))
        cfg = ' '.join(f'{k}={env[k]}' for k in FIT_DEFAULTS)
        print(f'[{mode} rep {rep}] fitting ({cfg}) -> {d / "fit.log"}', flush=True)
        t0 = time.time()
        with open(d / 'fit.log', 'w') as log:
            r = subprocess.run([sys.executable, '-u', str(HERE / 'fit_hierarchical.py')],
                               env=env, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)
        status = 'done' if r.returncode == 0 else f'FAILED (exit {r.returncode})'
        print(f'[{mode} rep {rep}] {status} in {(time.time() - t0) / 60:.1f} min', flush=True)
        if r.returncode:
            raise SystemExit(f'fit failed; see {d / "fit.log"}')


# -------------------------------------------------------------- analyze ----

def _ci(x, axis=0):
    return np.percentile(x, [2.5, 97.5], axis=axis)


def first_commit(O):
    """(..., T) order trajectories -> first trial with a committed order (T if never)."""
    comm = O != NB.UNK_O
    return np.where(comm.any(-1), comm.argmax(-1), O.shape[-1])


def typology_means(nat_mu):
    """(S, W) natural-scale condition means -> {group: (S,)}."""
    return {g: nat_mu[:, idx].mean(-1) for g, idx in RP.TYPOLOGY.items()}


def analyze_rep(mode, rep):
    sim = np.load(sim_path(mode, rep))
    fit = HLP.load_pickle(HLP.FIT_NAME, rep_dir(mode, rep))
    mu = HLP.pool_hyper(fit, 'mu')                       # (S, W, K)
    sigma, tau = HLP.pool_hyper(fit, 'sigma'), HLP.pool_hyper(fit, 'tau')
    tpar = HLP.pool_chains(fit, 'tpar')                  # (S, P, K)
    O = HLP.pool_chains(fit, 'O')                        # (S, P, T)
    B = HLP.pool_chains(fit, 'B')                        # (S, P, T, 7)
    r = dict(rep=rep, n_samples=mu.shape[0], n_chains=len(fit['chains']))
    r['mu_true'], r['mu_med'], r['mu_ci'] = sim['true_mu'], np.median(mu, 0), _ci(mu)
    r['tpar_true'], r['tpar_mean'], r['tpar_ci'] = sim['true_tpar'], tpar.mean(0), _ci(tpar)
    r['sigma_true'], r['sigma_ci'] = sim['true_sigma'], _ci(sigma)
    r['tau_true'], r['tau_ci'] = sim['true_tau'], _ci(tau)
    # typological p_commit contrasts, true and posterior
    gm = typology_means(NB.tpar_to_nat_np(mu[:, :, I_COMMIT], I_COMMIT))
    gm_true = typology_means(NB.tpar_to_nat_np(sim['true_mu'][None, :, I_COMMIT], I_COMMIT))
    r['contrast'] = {f'{a}/{b}': (float(gm_true[a][0] / gm_true[b][0]),
                                 gm[a] / gm[b]) for a, b in CONTRASTS}
    # latents
    O_true, B_true = sim['true_O'], sim['true_B']
    r['p_true_order'] = (O == O_true[None]).mean(0)                        # (P, T)
    r['p_true_belief'] = (B == B_true[None]).mean(0)                       # (P, T, 7)
    r['commit_true'] = first_commit(O_true)                                 # (P,)
    r['commit_post'] = np.median(first_commit(O), 0)                        # (P,)
    r['commit_ci'] = _ci(first_commit(O))                                   # (2, P)
    r['acc'] = fit['acc']
    return r


def analyze(mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reps = [r for r in reps_available(mode) if (rep_dir(mode, r) / HLP.FIT_NAME).exists()]
    if not reps:
        raise SystemExit(f'no fitted replicates under {PR_DIR / mode}')
    res = [analyze_rep(mode, r) for r in reps]
    R = len(res)
    names = CFG.PARAM_NAMES
    print(f'\n=== parameter recovery, mode "{mode}": {R} replicate datasets, '
          f'{res[0]["n_chains"]} chains x {res[0]["n_samples"] // res[0]["n_chains"]} '
          f'samples each ===')

    # ---- condition means ----
    mu_true = np.stack([r['mu_true'] for r in res])          # (R, W, K)
    mu_med = np.stack([r['mu_med'] for r in res])
    mu_lo = np.stack([r['mu_ci'][0] for r in res])
    mu_hi = np.stack([r['mu_ci'][1] for r in res])
    cover_mu = (mu_lo <= mu_true) & (mu_true <= mu_hi)
    print('\ncondition means mu[w, k] (transformed scale), pooled over replicates and conditions:')
    print(f'  {"param":>17}  {"95% cover":>9}  {"bias":>7}  {"RMSE":>6}  {"CI width":>8}  {"true sd":>7}')
    for k, n in enumerate(names):
        e = mu_med[:, :, k] - mu_true[:, :, k]
        print(f'  {n:>17}  {cover_mu[:, :, k].mean():>7.0%}    {e.mean():>+7.3f}  '
              f'{np.sqrt((e ** 2).mean()):>6.3f}  {(mu_hi - mu_lo)[:, :, k].mean():>8.3f}  '
              f'{mu_true[:, :, k].std():>7.3f}')

    # ---- per-participant tpar ----
    t_true = np.concatenate([r['tpar_true'] for r in res])   # (R*P, K)
    t_mean = np.concatenate([r['tpar_mean'] for r in res])
    t_lo = np.concatenate([r['tpar_ci'][0] for r in res])
    t_hi = np.concatenate([r['tpar_ci'][1] for r in res])
    cover_t = (t_lo <= t_true) & (t_true <= t_hi)
    print('\nper-participant tpar[p, k] (transformed scale), pooled over replicates:')
    print(f'  {"param":>17}  {"95% cover":>9}  {"bias":>7}  {"RMSE":>6}  {"corr":>5}  '
          f'{"CI width":>8}  {"true sd":>7}')
    for k, n in enumerate(names):
        e = t_mean[:, k] - t_true[:, k]
        c = np.corrcoef(t_true[:, k], t_mean[:, k])[0, 1]
        print(f'  {n:>17}  {cover_t[:, k].mean():>7.0%}    {e.mean():>+7.3f}  '
              f'{np.sqrt((e ** 2).mean()):>6.3f}  {c:>5.2f}  {(t_hi - t_lo)[:, k].mean():>8.3f}  '
              f'{t_true[:, k].std():>7.3f}')

    # ---- hyper sds ----
    print('\nhyper sds (95% coverage over replicates; truth vs posterior interval of rep 0):')
    for key in ('sigma', 'tau'):
        tr = np.stack([r[f'{key}_true'] for r in res])
        lo = np.stack([r[f'{key}_ci'][0] for r in res])
        hi = np.stack([r[f'{key}_ci'][1] for r in res])
        cov = ((lo <= tr) & (tr <= hi)).mean(0)
        print(f'  {key:>6}: ' + '  '.join(f'{names[k]} {cov[k]:.0%} ({tr[0, k]:.2f} in '
                                          f'[{lo[0, k]:.2f}, {hi[0, k]:.2f}])' for k in range(K)))

    # ---- contrasts ----
    print('\ntypological p_commit contrasts (ratio of group means): true | posterior '
          'median [95% CI] P(>1), per replicate:')
    sig_any = 0
    for r in res:
        line = f'  rep {r["rep"]:2d}: '
        any_sig = False
        for name, (tr, post) in r['contrast'].items():
            lo, hi = _ci(post)
            pgt = (post > 1).mean()
            any_sig |= pgt > 0.975 or pgt < 0.025
            line += f'{name} {tr:5.2f} | {np.median(post):5.2f} [{lo:5.2f}, {hi:5.2f}] {pgt:.3f}   '
        sig_any += any_sig
        print(line)
    if mode == 'null':
        print(f'  replicates with a "significant" contrast (P(>1) outside [0.025, 0.975]) '
              f'although all condition means are equal: {sig_any}/{R}')

    # ---- latents ----
    p_o = np.concatenate([r['p_true_order'] for r in res])          # (R*P, T)
    p_b = np.concatenate([r['p_true_belief'] for r in res])         # (R*P, T, 7)
    c_true = np.concatenate([r['commit_true'] for r in res])
    c_post = np.concatenate([r['commit_post'] for r in res])
    c_lo = np.concatenate([r['commit_ci'][0] for r in res])
    c_hi = np.concatenate([r['commit_ci'][1] for r in res])
    ever = c_true < T
    print('\nlatent trajectories:')
    print(f'  posterior P(true order) per trial: mean {p_o.mean():.3f} '
          f'(trials before the true commit {p_o[np.arange(T)[None] < c_true[:, None]].mean():.3f}, '
          f'after {p_o[np.arange(T)[None] >= c_true[:, None]].mean():.3f})')
    print(f'  posterior P(true belief) per (trial, word): mean {p_b.mean():.3f}')
    if ever.any():
        e = c_post[ever] - c_true[ever]
        cov = ((c_lo <= c_true) & (c_true <= c_hi))[ever].mean()
        print(f'  commit time (ppts that commit by trial {T}: {ever.mean():.0%}): '
              f'median abs error {np.median(np.abs(e)):.0f} trials, bias {e.mean():+.1f}, '
              f'95% coverage {cov:.0%}, corr {np.corrcoef(c_true[ever], c_post[ever])[0, 1]:.2f}')

    # ---- figure ----
    fig, axes = plt.subplots(3, K, figsize=(4.2 * K, 12))
    log_k = [n in ('logit_eps_o', 'logit_p_commit_o') for n in names]
    for k, n in enumerate(names):
        ax = axes[0, k]
        for w in range(W):
            x = NB.tpar_to_nat_np(mu_true[:, w, k], k)
            y = NB.tpar_to_nat_np(mu_med[:, w, k], k)
            lo = NB.tpar_to_nat_np(mu_lo[:, w, k], k)
            hi = NB.tpar_to_nat_np(mu_hi[:, w, k], k)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt='o', ms=4, color=RP.order_colors[w],
                        label=RP.order_labels[w], alpha=0.8, lw=0.8)
        allv = np.concatenate([NB.tpar_to_nat_np(mu_true[:, :, k], k).ravel(),
                               NB.tpar_to_nat_np(mu_hi[:, :, k], k).ravel(),
                               NB.tpar_to_nat_np(mu_lo[:, :, k], k).ravel()])
        ax.plot([allv.min(), allv.max()], [allv.min(), allv.max()], 'k--', lw=0.7)
        if log_k[k]:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{n}: condition means\n95% coverage {cover_mu[:, :, k].mean():.0%}',
                     fontsize=9)
        ax.set_xlabel('true (natural scale)'); ax.set_ylabel('posterior median [95% CI]')
        if k == 0:
            ax.legend(fontsize=7)
        ax = axes[1, k]
        ax.scatter(t_true[:, k], t_mean[:, k], s=4, alpha=0.3, c='#377eb8')
        lim = [min(t_true[:, k].min(), t_mean[:, k].min()), max(t_true[:, k].max(), t_mean[:, k].max())]
        ax.plot(lim, lim, 'k--', lw=0.7)
        ax.set_title(f'{n}: per-participant\ncover {cover_t[:, k].mean():.0%}, '
                     f'r = {np.corrcoef(t_true[:, k], t_mean[:, k])[0, 1]:.2f}', fontsize=9)
        ax.set_xlabel('true (transformed)'); ax.set_ylabel('posterior mean')
    ax = axes[2, 0]
    for i, (name, _) in enumerate(res[0]['contrast'].items()):
        tr = [r['contrast'][name][0] for r in res]
        med = [np.median(r['contrast'][name][1]) for r in res]
        lo = [_ci(r['contrast'][name][1])[0] for r in res]
        hi = [_ci(r['contrast'][name][1])[1] for r in res]
        ax.errorbar(tr, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt='o', ms=4,
                    lw=0.8, alpha=0.8, label=name, color=['#1b9e77', '#d95f02', '#7570b3'][i])
    ax.set_xscale('log'); ax.set_yscale('log')
    lim = ax.get_xlim(); ax.plot(lim, lim, 'k--', lw=0.7)
    ax.axhline(1, color='#888', lw=0.5); ax.axvline(1, color='#888', lw=0.5)
    ax.set_xlabel('true ratio'); ax.set_ylabel('posterior median [95% CI]')
    ax.set_title('p_commit typological contrasts', fontsize=9); ax.legend(fontsize=7)
    ax = axes[2, 1]
    ax.scatter(c_true[ever], c_post[ever], s=5, alpha=0.4, c='#e6550d')
    ax.plot([0, T], [0, T], 'k--', lw=0.7)
    ax.set_xlabel('true commit trial'); ax.set_ylabel('posterior median commit trial')
    ax.set_title(f'order commit time ({ever.sum()} committing ppts)', fontsize=9)
    ax = axes[2, 2]
    ax.plot(p_o.mean(0), color='#377eb8', label='order')
    ax.plot(p_b.mean((0, 2)), color='#4daf4a', label='belief (per word)')
    ax.set_ylim(0, 1.02); ax.set_xlabel('trial'); ax.set_ylabel('posterior P(true latent)')
    ax.set_title('latent recovery by trial', fontsize=9); ax.legend(fontsize=8)
    ax = axes[2, 3]
    acc = np.concatenate([r['acc'] for r in res])
    ax.scatter(acc, p_o.mean(1), s=5, alpha=0.4, c='#377eb8', label='order')
    ax.scatter(acc, p_b.mean((1, 2)), s=5, alpha=0.4, c='#4daf4a', label='belief')
    ax.set_xlabel('simulated accuracy (vs true scene)'); ax.set_ylabel('mean posterior P(true latent)')
    ax.set_title('latent recovery vs accuracy', fontsize=9); ax.legend(fontsize=8)
    fig.suptitle(f'Parameter recovery, mode "{mode}" ({R} replicates)', fontsize=11)
    plt.tight_layout()
    HLP.savefig(fig, f'param_recovery_{mode}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['simulate', 'fit', 'analyze'])
    ap.add_argument('--mode', choices=MODES, default='prior')
    ap.add_argument('--n_rep', type=int, default=8, help='simulate: number of datasets')
    ap.add_argument('--seed', type=int, default=2026, help='simulate: base seed')
    ap.add_argument('--rep', type=int, nargs='*', help='fit: replicate index(es); default all')
    ap.add_argument('--force', action='store_true', help='fit: refit existing replicates')
    a = ap.parse_args()
    if a.cmd == 'simulate':
        simulate(a.mode, a.n_rep, a.seed)
    elif a.cmd == 'fit':
        fit(a.mode, a.rep if a.rep else reps_available(a.mode), a.force)
    else:
        analyze(a.mode)
