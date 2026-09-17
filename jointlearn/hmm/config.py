"""
Prior / initialisation config shared by 01_fit.py and the
downstream scripts. The transformed-scale parameter layout is documented
in model.to_nat.

Hierarchy, per transformed parameter k, with w(p) the word-order condition
(0..5 = SVO, SOV, VSO, VOS, OSV, OVS) of participant p:

    tpar[p, k]  ~ N(mu[w(p), k], sigma_k^2)        participant within condition
    mu[w, k]    ~ N(m_k, tau_k^2)                  condition mean
    m_k         ~ N(PRIOR_MU0[k], PRIOR_SD0[k]^2)  population mean
    sigma_k^2   ~ InvGamma(A0, B0)                 within-condition spread
    tau_k^2     ~ InvGamma(A_TAU, B_TAU)           between-condition spread

All updates are conjugate (see sampler.update_hyper). Condition
differences in learnability are read off the posterior of mu[:, k]
(03_condition_effects.py); tau_k measures how much the six condition
means differ.

The swap cost kappa_s is NOT fitted (it is unidentified — see README and
model.KAPPA_S_FIXED), so there are 4 fitted parameters.
"""
import numpy as np

# transformed params: [logit lam, log kappa, logit eps_o, logit p_commit_o]
PARAM_NAMES = ['logit_lam', 'log_kappa', 'logit_eps_o', 'logit_p_commit_o']
N_TPAR = len(PARAM_NAMES)

# initial (mu, sigma) per transformed param; used for all condition means
HYPER_INIT = np.array([[np.log(0.08 / 0.92), 0.7],
                       [np.log(4.5), 0.4],
                       [np.log(0.01 / 0.99), 0.7],
                       [np.log(0.03 / 0.97), 0.6]])
TAU_INIT = 0.3          # initial between-condition sd (all params)

# population-mean prior m_k ~ N(PRIOR_MU0[k], PRIOR_SD0[k]^2)
PRIOR_MU0 = np.array([
    np.log(0.03 / 0.97),   # logit lambda  — anchor at lambda = 0.03 (small lapse)
    np.log(4.5),           # log kappa
    np.log(0.01 / 0.99),   # logit eps_o
    np.log(0.03 / 0.97),   # logit p_commit_o
])
# sd 0.5 on the logit scale = a mild pull of the population lapse rate
# toward 0.03 (roughly lambda in [0.011, 0.076] at +-1 sd); the other three
# are effectively flat.
PRIOR_SD0 = np.array([0.5, 3.0, 3.0, 3.0])

# Random-walk proposal sd per transformed coordinate for the per-participant
# coordinate-wise Metropolis step (sampler._mh4_one). Set to ~2.4x
# the median per-participant posterior sd measured on the 2026-09-12 fit
# (0.45, 0.08, 0.67, 1.23); coordinate-wise random walk is near-optimal at
# ~0.44 acceptance. Part of the checkpoint config tuple.
PROP_SD = np.array([1.0, 0.2, 1.6, 3.0])

# Proposal sd per coordinate for the group shift move (sampler.
# shift_all): one delta added to a condition mean AND to every participant
# in that condition. Tuned on the 2026-09-12 fit state for ~0.3-0.5
# acceptance (see README). Part of the checkpoint config tuple.
SHIFT_SD = np.array([0.15, 0.03, 1.0, 0.5])
# Same for the population-level shift (m, all mu[w] and all tpar together).
SHIFT_SD_POP = np.array([0.06, 0.01, 1.2, 0.2])

A0, B0 = 2.0, 1.0        # sigma_k^2 ~ InvGamma(2, 1): prior mean 1
A_TAU, B_TAU = 2.0, 0.5  # tau_k^2 ~ InvGamma(2, 0.5): prior mean 0.5 (tau ~ 0.7)
