import pandas as pd
import numpy as np
import seaborn as sns
import pymc3 as pm
import matplotlib.pyplot as plt
import os


def plot_simple_logistic_by_condition(trace, analysis_arrays):
    
    data_df = (
        trace
        .observed_data
        .chosen_scenes
        .to_dataframe()
        .reset_index()
    )

    # add the word order for each participant
    data_df.loc[:,'orders'] = (
        analysis_arrays
        ['word_order_partic']
        .values
        [data_df['participant']]
    )
    # add whether response was correct
    data_df.loc[:,'right'] = data_df['chosen_scenes']==0

    sns.lmplot(
        data=data_df,
        x='trial',
        y='right',
        hue='orders',
        logistic=True,
        scatter_kws={
            'marker': '|'
        }
    )
                                      
    plt.show()
    
    
def plot_posterior_predictive(model, trace, analysis_arrays, 
                              posterior_pred=None, folderp=None):
    
    data_df = (
        trace
        .observed_data
        .chosen_scenes
        .to_dataframe()
        .reset_index()
    )
    
    # add the word order for each participant
    data_df.loc[:,'orders'] = (
        analysis_arrays
        ['word_order_partic']
        .values[data_df['participant']]
    )
    # add whether response was correct
    data_df.loc[:,'right'] = (
        data_df
        ['chosen_scenes']
        ==0
    )

    if posterior_pred is None:
        with model:
            # simulate experiments, and specifically
            # whether a participants gets the answers
            # right in each trial
            posterior_pred = pm.sample_posterior_predictive(
                trace
            )

    for i in range(data_df['participant'].max()+1):
        
        filename = folderp+f"part-{i}.png"
        
        if os.path.isfile(filename):
            print(f'{filename} exists, going to next')
            continue
            
        predictions = posterior_pred['chosen_scenes'][:,:,i]
        actual = data_df[data_df['participant']==i]

        sample, trial = np.indices(predictions.shape)
        right = actual['right'].values[trial.flatten()]
        sim_df = pd.DataFrame({
            'sim_right': (predictions==0).flatten(),
            'actual_right': right.flatten(),
            'trial': trial.flatten()
        })
        fig, ax = plt.subplots(figsize=(8,3))

        sns.pointplot(
            data=sim_df,
            x='trial',
            y='sim_right',
            join=False,
            hue='actual_right',
            scale=0.5,
            errwidth=1,
            capsize=0.5,
            palette={True: 'green', False: 'red'},
            legend=False,
            ax=ax
        )
        ax.legend().set_visible(False)

        sns.regplot(
            data=actual,
            x='trial',
            y='right',
            marker='|',
            logistic=True,
            ax=ax
        )
        
        x = analysis_arrays['word_order_partic'].values[i]
        ax.set_title(
            f'Participant {i}, order {x}'
        )
        ax.set_ylim(0,1)
        ax.set_xticks(
            np.arange(0, predictions.shape[1], 10)
        )
        if folderp is None:
            plt.show()
        else:
            fig.savefig(
                filename,
                dpi=300
            )
        
            
    return posterior_pred


def log_mean_exp(arr,axis=None):
    
    arr_max = np.max(
        arr, 
        axis=axis, 
        keepdims=True
    )

    mean = (
        np.log(
            np.mean(
                np.exp(arr - arr_max), 
                axis=axis, 
                keepdims=True
            )
        ) + arr_max
    )
    return mean 
    
    
def calculate_loglikmeans(trace_free, trace_weight):

    # get the loglikelihoods of the data
    # for each posterior sample
    logliks_free = (
        trace_free
        .log_likelihood
        .right
        .values
        .squeeze()
    )
        
    # logliks_weight = (
    #     trace_weight
    #     .log_likelihood
    #     .chosen_scenes
    #     .values
    #     .reshape(-1, logliks_free.shape[-1])
    # )
    
    try:
        # loglikelihood of correct answer for weight model
        logliks_weights = (
            trace_weight
            .posterior
            .p_correct
            # Dimensions (chain, sample, trial, participant)
            .values
            .squeeze()
        )
    
    except AttributeError:
        print("The trace doesn't have p correct for the weight model!")
        return
    
    logliks_weights = logliks_weights.reshape(
        len(logliks_weights), 
        -1
    )

#     loglikmean_weight = log_mean_exp(
#         logliks_weights,
#         0
#     ).flatten()
    
    #### Calculate loglikelihoods of observed data

    loglikmean_weight_observed = np.log(
        # this is the probability of the participant
        # choosing the correct answer or not
        np.where(
            trace_free.observed_data.right.values,
            # when the answer was correct, select prob of choosing
            logliks_weights,
            # when the answer was incorrect, select prob of not choosing
            1-logliks_weights
        ).mean(0)
    )
    
    # Calculate the log of the mean posterior probability
    # of the participant getting answer correct/incorrect
    # Only taking first thousand samples for free model 
    # because running function on whole posterior trace
    # crashed the kernel
    loglikmean_free_observed = log_mean_exp(
        logliks_free[:1000],
        0
    ).flatten()
    

    # loglikmean_free_observed = np.log(
    #     np.where(
    #         trace_free.observed_data.right.values,
    #         np.exp(logliks_free[:1000]),
    #         1-np.exp(logliks_free[:1000])
    #     )
    # )
    
    return loglikmean_free_observed, loglikmean_weight_observed
    

def plot_participantwise_loglik_differences(analysis_arrays, filename,
                                            trace_free=None, trace_weight=None, 
                                            loglikmean_free_observed=None, loglikmean_weight_observed=None,
                                            plot_type='by_difference'):
    
    assert (
        ((trace_free is not None) and (trace_weight is not None))
        or
        ((loglikmean_free_observed is not None) and (loglikmean_weight_observed is not None))
    ), "Please pass either the traces or the loglikmeans"
    
    if (loglikmean_free_observed is None) or (loglikmean_weight_observed is None):
        loglikmean_free_observed, loglikmean_weight_observed = calculate_loglikmeans(
            trace_free, 
            trace_weight
        )
    
    trial, partic = np.indices(
        analysis_arrays['history_choices_indices']
        .shape
    )
    
    trial = trial.flatten()
    partic = partic.flatten()

    fig, axes = plt.subplots(
        (partic.max()//10)+1, 
        10, 
        figsize=(20, 60)
    )

    axes_flat = axes.flatten()
    
    if plot_type == "by_difference":
        
        for i in range(partic.max()):
            indices = partic == i
            ax = axes_flat[i]
            
            ax.scatter(
                # x axis has mean loglik of free-model
                x=loglikmean_free_observed[indices],
                # y axis has weight model
                y=loglikmean_weight_observed[indices],
                s=0.5,
                alpha=0.5,
                c=trial[indices].astype(float),
            )

            ax.axline(
                [0,0], 
                [1,1], 
                ls="--", 
                c=".1", 
                # transform=ax.transAxes
            )

            ax.text(
                0.1, 0.8, i,
                transform=ax.transAxes
            )
            
            print(i, end=' ')

        [
            ax.set_xlabel('Logistic regression') 
            for ax in axes[-1,:]
        ]

        [
            ax.set_ylabel('Weight model') 
            for ax in axes[:,0]
        ]

    elif plot_type == "by_trial":

        for i in range(partic.max()):
            
            indices = partic == i
            ax = axes_flat[i]
            ax.scatter(
                x=loglikmean_free_observed[indices] - loglikmean_weight_observed[indices],
                y=trial[indices],
                s=0.5,
                alpha=0.5,
                # c=trial[indices].astype(float),
                c=trace_free.observed_data.right.values[indices]
            )
            
            print(i, end=' ')
            ax.axvline(0, color='red', ls='-.')
            
            ax.text(
                0.1, 0.8, i,
                transform=ax.transAxes
            )
            
        [
            ax.set_xlabel((
                'Logistic regression - '
                'weight model\n(Pointwise posterior log meanlikelihood)'
            )) 
            for ax in axes[-1,:]
        ]
        
        [
            ax.set_ylabel('Trial') 
            for ax in axes[:,0]
        ]

    else:
        raise InputError("Plot type not implemented!")

    plt.tight_layout()

    fig.savefig(
        filename, 
        dpi=300
    )
    plt.close(fig)