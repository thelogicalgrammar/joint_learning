import bambi as bmb
import pandas as pd
import numpy as np
import arviz as az

def analyse(history_choices_indices, word_order_partic):
    
    # the right choice is always 0
    # Simplify to just whether the participant gets it right
    # (Without further modelling structure the other three options are
    # not structured)
    
    right_answers = history_choices_indices == 0

    trial_index, participant_index = np.indices(history_choices_indices.shape)

    bmb_df = pd.DataFrame(
        data={
            'trial': trial_index.flatten(),
            'partic': participant_index.flatten(),
            # word orders
            'order': word_order_partic.values[participant_index.flatten()],
            # signals,
            'right': right_answers.flatten()
        }
    )

    bmb_df.loc[:,'order_id'], order_codes = pd.factorize(
        bmb_df['order']
    )

    bmb_df.loc[:, 'scaled_trial'] = bmb_df.trial / bmb_df.trial.max()

    np.unique(bmb_df[bmb_df['trial']==2]['order_id'], return_counts=True)

    bmb_df.isna().any(axis=1).sum()

    model = bmb.Model(
         "right ~ scaled_trial + order_id + (1|partic)",
        family='bernoulli',
        link='logit',
        categorical=['order_id', 'partic'],
        data=bmb_df
    )
    
    model.build()

    try:
        results = az.from_netcdf(
            'results/trace_modelfree.cdf'
        )
    except FileNotFoundError:
        
        results = model.fit(
            chains=1, 
            draws=3000
        )
        # predict datapoints on the probability scale
        # model.predict(results)
        az.to_netcdf(results, 'results/trace_modelfree.cdf')
    
    bmb_df.loc[:,'predicted_mu'] = results.posterior.right_mean.mean(axis=(0,1))
    bmb_df.loc[:,'predicted_var'] = results.posterior.right_mean.var(axis=(0,1))
    
    return model, results, bmb_df