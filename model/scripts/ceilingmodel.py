try:
    from model.scripts.data_functions import get_data, get_analysis_arrays
except ImportError:
    from scripts.data_functions import get_data, get_analysis_arrays

import numpy as np
import pandas as pd
import pymc as pm

from os import makedirs
import pickle
import argparse

def define_model(df, include_p_row=False):
    
    # numeric participant and order codes
    pid_codes,  pid_idx   = np.unique(df.partic.values, return_inverse=True)
    ord_codes,  ord_idx_t = np.unique(df.order .values, return_inverse=True)
    P  = pid_codes.size                       # participants
    K  = ord_codes.size                       # order levels

    # For the order main effect we need one code *per participant*
    order_per_pid = np.zeros(P, dtype=int)
    order_per_pid[np.arange(P)] = ord_idx_t[np.searchsorted(pid_idx, np.arange(P))]

    # trial indices per row 
    t        = df.trial.values.astype("float32")
    pid_row  = pid_idx.astype("int32")
    y_obs    = df.correct.values.astype("int8")

    with pm.Model() as cascade:

        # hyper-priors (weakly informative) 

        # ---------- hyper-priors (weakly informative) --------------------------
        #           centre each link scale around 0 with broad SD = 1
        mu_log_s     = pm.Normal("mu_log_s",   0., 2.)
        mu_log_T     = pm.Normal("mu_log_T",   100, 50.)
        mu_logit_C   = pm.Normal("mu_logit_C", pm.math.logit(0.9), 0.3)

        # variation between participants within their word order
        sigma_log_s  = pm.HalfNormal("sigma_log_s", 2.)
        sigma_log_T  = pm.HalfNormal("sigma_log_T", 10.)
        sigma_logitC = pm.HalfNormal("sigma_logitC", 0.3)

        # variation of the word order effects around their mean,
        # with a sum-to-zero constraint
        # soft-centre them so the group mean is the intercept μ
        ord_s  = pm.Normal("ord_s",  0., 0.5, shape=K)
        ord_T  = pm.Normal("ord_T",  0., 20., shape=K)
        ord_C  = pm.Normal("ord_C",  0., 1., shape=K)
        ord_s  = ord_s  - pm.math.mean(ord_s)
        ord_T  = ord_T  - pm.math.mean(ord_T)
        ord_C  = ord_C  - pm.math.mean(ord_C)

        # participant-level values in unconstrained spaces
        log_s_i    = pm.Normal("log_s_i",
                            mu_log_s   + ord_s[order_per_pid],
                            sigma_log_s,
                            shape=P)
        log_T_i    = pm.Normal("log_T_i",
                            mu_log_T   + ord_T[order_per_pid],
                            sigma_log_T,
                            shape=P)
        logit_C_i  = pm.Normal("logit_C_i",
                            mu_logit_C + ord_C[order_per_pid],
                            sigma_logitC,
                            shape=P)

        # ---------- transform to the natural scales ----------------------------
        # enforce s_i>1 by adding 1 after the softplus
        s_i = pm.Deterministic("s_i", pm.math.log1pexp(log_s_i)+1)
        T_i = pm.Deterministic("T_i", pm.math.log1pexp(log_T_i))     # >0
        C_i = pm.Deterministic("C_i", pm.math.sigmoid(logit_C_i))    # in (0,1)

        # vectorised probability for every row 
        # gather participant parameters for each trial row
        s_row = s_i[pid_row]
        T_row = T_i[pid_row]
        C_row = C_i[pid_row]

        ratio  = t / T_row
        grow   = 0.25 + (C_row - 0.25) * ratio**s_row
        p_row  = pm.math.switch(t < T_row, grow, C_row)

        if include_p_row:
            pm.Deterministic('p_row', p_row)

        # likelihood 
        pm.Bernoulli("y", p_row, observed=y_obs)

    return cascade


def prior_predictive(df):
    
    cascade = define_model(df)

    with cascade:
        trace = pm.sample(
            draws       = 1000,
            tune        = 1000,
            cores       = 16,
            chains      = 16,
            mp_ctx      ="spawn"
        )

        trace.to_netcdf('./results/cascademodel.cdf')


def parameter_recovery(df, jobindex):

    cascade = define_model(df)

    advi_samples = 10000
    posterior_samples = 1000

    with cascade:
        prior_pred = pm.sample_prior_predictive(samples=1)

    y = prior_pred.prior_predictive['y'].squeeze()
    new_df = df.copy()
    new_df['correct'] = y
    cascade_aux = define_model(new_df, True)
    with cascade_aux:
        meanfield = pm.fit(n=advi_samples, method='advi', progressbar=True)

    samples = {}
    for var in [
                'mu_log_T', 'mu_log_s', 'mu_logit_C',
                'sigma_log_T', 'sigma_log_s', 'sigma_logitC',
                'ord_C', 'ord_s', 'ord_T'
        ]:
        samples[var] = meanfield.sample_node(vars(cascade_aux)[var], posterior_samples).eval()

    # create folder
    makedirs('parameter_recovery', exist_ok=False)

    prior_pred.to_netcdf(f'parameter_recovery/prior_pred_{jobindex}.nc')
    with open(f'parameter_recovery/pred_samples_{jobindex}.json', 'wb') as f:
        pickle.dump(samples, f)


if __name__ == "__main__":

    # take command line argument for whether to do prior predictive sampling
    # or parameter recovery
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_predictive', action='store_true')
    parser.add_argument('--jobindex', type=int, default=0)
    args = parser.parse_args()

    # run from main folder
    data = get_data("../data.csv")
    analysis_arrays = get_analysis_arrays(data)

    word_order_partic = analysis_arrays['word_order_partic'] 
    history_choices_indices = analysis_arrays['history_choices_indices']

    right_answers = (history_choices_indices == 0).astype(int)

    trial_index, participant_index = np.indices(
        history_choices_indices.shape
    )

    df = pd.DataFrame(
        data={
            'trial': trial_index.flatten(),
            'partic': participant_index.flatten(),
            # word orders
            'order': word_order_partic.values[
                participant_index.flatten()
            ],
            # signals,
            'correct': right_answers.flatten()
        }
    )

    if args.prior_predictive:
        prior_predictive(df)
    else:
        parameter_recovery(df, args.jobindex)
