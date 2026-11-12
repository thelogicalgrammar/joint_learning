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

Sampled jointly by HMC (non-centred) within Gibbs, see sampler.py. Condition
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

# initial (mu, sigma) per transformed param: sigma is the centre of the
# chains' dispersed starting values (sampler.start_values); mu is used by the tests
HYPER_INIT = np.array([[np.log(0.08 / 0.92), 0.7],
                       [np.log(4.5), 0.4],
                       [np.log(0.01 / 0.99), 0.7],
                       [np.log(0.03 / 0.97), 0.6]])
TAU_INIT = 0.3          # centre of the starting between-condition sd (all params)

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

A0, B0 = 2.0, 1.0        # sigma_k^2 ~ InvGamma(2, 1): prior mean 1
A_TAU, B_TAU = 2.0, 0.5  # tau_k^2 ~ InvGamma(2, 0.5): prior mean 0.5 (tau ~ 0.7)
