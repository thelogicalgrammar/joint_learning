import bambi as bmb
import pandas as pd
import numpy as np
import arviz as az

from .data_functions import get_data, get_first_n_trials, get_analysis_arrays

def analyse(participant_exclusion=True, first_n_trials='all',
            fit_kwargs=None):
    
    data = get_data()

    analysis_arrays = get_analysis_arrays(
        data,
        exclude_nonimproving_participants=participant_exclusion
    )
    
    history_choices_indices = analysis_arrays['history_choices_indices']
    word_order_partic = analysis_arrays['word_order_partic']
    
    print('Got and wrangled the data')
    
    if first_n_trials is not None:
        print(f'Getting only the first {first_n_trials} trials')
        analysis_arrays = get_first_n_trials(
            analysis_arrays,
            n_trials=first_n_trials
        )

    outputfile_name = (
        'results/'
        'modelfree'
        f'_excluded-{participant_exclusion}'
        f'_trialsupto-{first_n_trials}'
        '.cdf'
    )
    
    print(f"Going to save with name {outputfile_name}")
    # the right choice is always 0
    # Simplify to just whether the participant gets it right
    # (Without further modelling structure the other three options are
    # not structured)
    
    right_answers = history_choices_indices == 0

    trial_index, participant_index = np.indices(
        history_choices_indices.shape
    )

    bmb_df = pd.DataFrame(
        data={
            'trial': trial_index.flatten(),
            'partic': participant_index.flatten(),
            # word orders
            'order': word_order_partic.values[
                participant_index.flatten()
            ],
            # signals,
            'right': right_answers.flatten()
        }
    )

    bmb_df.loc[:,'order_id'], order_codes = pd.factorize(
        bmb_df['order']
    )

    bmb_df.loc[:, 'scaled_trial'] = bmb_df.trial / bmb_df.trial.max()

    # np.unique(bmb_df[bmb_df['trial']==2]['order_id'], return_counts=True)

    print(f'There are {bmb_df.isna().any(axis=1).sum()} NaNs in the data')

    model = bmb.Model(
         "right ~ scaled_trial + order_id + (1|partic)",
        family='bernoulli',
        link='logit',
        categorical=['order_id', 'partic'],
        data=bmb_df
    )
    
    model.build()

    try:
        trace = az.from_netcdf(
            outputfile_name
        )
        print("Already found a file with that name, getting from file")

    except FileNotFoundError:
        
        if fit_kwargs is None:
            fit_kwargs = {
                'chains': 1, 
                'draws': 3000
            }
        trace = model.fit(
            **fit_kwargs
        )
        
        # predict datapoints on the probability scale
        print('Predicting datapoints on probability scale')
        model.predict(trace)
        
        az.to_netcdf(
            trace, 
            outputfile_name
        )
        print('Saved samples in results folder')
    
    try:
        trace.posterior.right_mean
    
    except AttributeError:
        
        print((
            "Trace doesn't contain predictions on prob scale, "
            "calculating and resaving trace"
        ))
        
        model.predict(trace)
        
        az.to_netcdf(
            trace, 
            outputfile_name+'recalculated'
        )
        print((
            'Saved samples in results folder'
            f'with name {outputfile_name}recalculated'
            ', change name back!'
        ))
        
    finally:
        
        bmb_df.loc[:,'predicted_mu'] = (
            trace
            .posterior
            .right_mean
            .mean(axis=(0,1))
        )

        bmb_df.loc[:,'predicted_var'] = (
            trace
            .posterior
            .right_mean
            .var(axis=(0,1))
        )
    
    return model, trace, bmb_df

if __name__=='__main__':
    
    analyse(
        participant_exclusion=False, 
        first_n_trials='all'
    )
