import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pymc3 as pm
import arviz as az
import theano 
from theano import tensor as tt
tprint = theano.printing.Print

from .simulation_functions import define_objects, normalize, simulate_full_experiment, define_objects
from .data_functions import get_data, get_first_n_trials, get_analysis_arrays

############## Simulate data with numpy

def update_weights(interpretation_weights_original, word_order_weights_original, scene, signal):
    
    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )

    subj, act, obj = scene
    firstword, secondword, thirdword = signal
    
    interpretation_weights = interpretation_weights_original.copy()
    word_order_weights = word_order_weights_original.copy()

    ### Snippet A

    # get mask for the indices that I wrote above for word_order_weights
    mask_orders = (
        np.tile(word_orders[None], (3,1,1)) 
        == 
        np.array([0,1,2])[:,None,None]
    )
    # get the values from word_order_weights
    # to add to the various elements
    # rows: (subj, obj, act)
    # cols: (first word, second word, third word)
    values_to_add = word_order_weights @ mask_orders
        
    # add value to right locations
    interpretation_to_add = np.zeros_like(
        interpretation_weights
    )
    
    interpretation_to_add[
        np.tile(signal[None], (3,1)), 
        np.tile(scene.reshape(-1,1), (1,3))
    ] = values_to_add
        
    interpretation_weights = normalize(
        interpretation_weights + interpretation_to_add,
        axis=1
    )
    
    # Snippet B
    
    word_order_to_add = interpretation_weights_original[
        np.tile(signal[None], (6,1)), 
        scene[word_orders]
    ].prod(1)
    
    word_order_weights = normalize(
        word_order_weights_original + 
        word_order_to_add
    )
    
    return interpretation_weights, word_order_weights


##################### PyMC3 

def theano_normalize(tensor, axis):
    return tensor / tensor.sum(axis, keepdims=True)


###### (single participant)


def update_weights_theano(scene, signal, interpretation_weights_original, 
                          word_order_weights_original, word_orders):
    
    ### Snippet A

    # get mask for the indices that I wrote above 
    # where I described how to update the array
    # for word_order_weights
    mask_orders = tt.eq(
        tt.tile(word_orders, (3,1,1)), 
        np.array([0,1,2])[:,None,None]
    )
    
    # get the values from word_order_weights
    # to add to the various elements
    # rows: (subj, obj, act)
    # cols: (first word, second word, third word)
    values_to_add = tt.batched_dot(
        tt.tile(word_order_weights_original, (3,1)), 
        mask_orders
    )
        
    # add value to right locations
    interpretation_to_add = tt.set_subtensor(
        tt.zeros(interpretation_weights_original.shape)[
            tt.tile(signal, (3,1)), 
            tt.tile(scene.dimshuffle(0, 'x'), 3)
        ],
        values_to_add
    )
    
    # normalize last dimension
    interpretation_weights = theano_normalize(
        interpretation_weights_original + 
        interpretation_to_add,
        axis=-1
    )
    
    # Snippet B
    
    word_order_to_add = interpretation_weights_original[
        tt.tile(signal.dimshuffle('x', 0), (6,1)), 
        scene[word_orders]
    ].prod(1)
    
    word_order_weights = (
        word_order_weights_original + 
        word_order_to_add
    )
    
    word_order_weights = theano_normalize(
        word_order_weights,
        0
    )
    
    return interpretation_weights, word_order_weights


def probs_languages_to_probs_scenes(interpretation_weights, word_order_weights, 
                                    signals, scenes_trials, word_orders):
    """
    Finds probability of choosing each of the four scenes in `scenes`
    given the observed utterance and the current probabilities 
    of word-meaning association and word orders
    """
    
    # meaning that each word would be expressing
    # assuming each word order is true
    # dims (trial, scene in trial, word orders, 3), values: meanings
    meanings_trials_orders = scenes_trials[:,:,word_orders]
    
    return normalize(
        (
            # get the probability of the words observed
            # in each trial expressing each scene
            # assuming each word order in turn
            interpretation_weights[
                tt.arange(interpretation_weights.shape[0])[:,None,None,None],
                signals[:,None,None],
                meanings_trials_orders
            ]
            # get joint probability of the observed words
            # (for each trial, word order and scene)
            .prod(-1) 
            # multiply by probability of word orders
            * word_order_weights.dimshuffle(0, 'x', 1)
        # marginalize out word order
        # to get probs of observed words in scene
        ).sum(-1),
        1
    )

##### Note: copied from notebook, needs some reworking
def fit_model_single_participant():
    with pm.Model() as model:

        # dims (word, meaning)
        # Initial (fixed, not estimated) prior over
        # interpretations
        interpretation_weights = theano_normalize(
            tt.ones((7,7)),
            axis=1
        )

        # shape 6
        # Estimated Dirichlet prior over word orders
        word_order_weights = pm.Dirichlet(
            'word_order_prior', 
            [1]*6
        )

        # scan is theano's way to do looping.
        # Returns probs of each word-meaning connection
        # and each word order respectively
        (probs_interpretations, probs_word_orders), outputs_info = theano.scan(
            update_weights_theano,
            sequences=[
                theano.shared(scenes[true_scenes]),
                theano.shared(signals)
            ],
            outputs_info=[
                interpretation_weights,
                word_order_weights
            ],
            non_sequences=[
                word_orders
            ],
            profile=True,
            name='update weights'
        )

        # Calculate the probability of each choice
        # In each trial
        # dims (trial, scenes in trial)
        prob_scenes = probs_languages_to_probs_scenes(
            probs_interpretations, 
            probs_word_orders, 
            signals, 
            scenes[indices_scenes_trials], 
            word_orders
        )

        pm.Categorical(
            'chosen_scenes',
            p=prob_scenes,
            observed=history_choices_indices
        )

    with model:
        trace = pm.sample(
            return_inferencedata=True
        )

    az.plot_trace(trace, compact=False)
    plt.tight_layout()
    plt.show()

    with model:
        fit = pm.fit()

    with model:
        fit_samples = az.from_pymc3(fit.sample(5000))

    az.plot_trace(fit_samples, compact=False)
    plt.tight_layout()
    plt.show()
    
    
############### Multiple participants


def update_weights_theano_multiple_participants(scene, signal, 
                                                interpretation_weights_original, 
                                                word_order_weights_original, 
                                                word_orders, learning_weights):
    """
    NOTE: HERE BE DRAGONS
    NOTE after looking a couple months later: Oh boy!
    
    This function takes a description of a single trial index of the experiment
    as well as the participant's current posterior over 
    interpretations and word orders
    and returns the updated participants' posterior.
    
    NOTE: This function calculates everything for *all agents* for one trial
    So it might differ from functions above that are written for one participant.
    TODO: Is it possible to rewrite this to do all trials at once efficiently? 
          Maybe some clever cumulative operation?
    
    Parameters
    ----------
    scene: array
        dims (participant, 3)
        Description of scene that the signal describes.
        A single scene is described by (agent, action, patient)
    signal: array
        dims (participant, 3)
        The three columns report the indices of the signals
    interpretation_weights_original: array
        dims: (participant, signal, meaning)
        shape: (# participants, 7, 7)
        Each row is a the prob of the signal referring to each
        meaning. Gets updated in this function in light
        of evidence given by scene/signal combos.
    word_order_weights_original: array
        dims: (participant, word order)
        shape: (# participants, 6)
    word_orders: array
        dims: (word order,3)
        shape: (6,3)
        Word orders (see functions above for description)
    Returns
    -------
    tuple of arrays
        interpretation_weights: array
            Same shape as interpretation_weights_original
        word_order_weights: array
            Same shape as word_order_weights_original
    """
    
    n_parti = interpretation_weights_original.shape[0]
    
    ### Snippet A (update interpretation weights)

    # get one-hot vectors for the indices as described above
    # in which to add the calculated weights that should be added
    # Dimensions (participant, object in scene, word order, signal)
    # shape: (# participants, 3, 6, 3)
    mask_orders = tt.eq(
        tt.tile(word_orders, (n_parti,3,1,1)), 
        np.array([0,1,2])[None,:,None,None]
    )
    
    # Get the values from word_order_weights_original
    # to add to the various elements
    # using the one-hot vectors in mask_orders.
    # values_to_add: shape (# participants, 3, 3)
    # dims (participant, scene components, words in signal)
    # rows: (subj, obj, act)
    # cols: (first word, second word, third word)    
    values_to_add = (
        # get the weights compatible with each word order
        (word_order_weights_original[:,None,:,None] * mask_orders)
        # sum over word order dimension
        .sum(2)
    )
    
    # indicates which participant
    arg1 = tt.tile(tt.arange(n_parti).dimshuffle(0,'x','x'), (1,3,3))
    # indicates which signals (rows)
    arg2 = tt.tile(signal.dimshuffle(0,'x',1), (1,3,1))
    # indicates which meanings (columns)
    arg3 = tt.tile(scene.dimshuffle(0, 1,'x'), 3)
    
    # insert value to right locations
    # keep 0 everywhere else
    interpretation_to_add = tt.set_subtensor(
        tt.zeros(interpretation_weights_original.shape)[
            arg1, arg2, arg3
        ],
        values_to_add
    )
    
    interpretation_weights = theano_normalize(
        interpretation_weights_original 
        + learning_weights[:,None,None]*interpretation_to_add,
        axis=-1
    )
    
    # Snippet B (update word order weights)
    
    word_order_to_add = interpretation_weights_original[
        tt.tile(tt.arange(n_parti).dimshuffle(0,'x','x'), (1,6,3)),
        # for each participant, repeat signal once
        # for each word order.
        # dims (participant, word order, signal)
        tt.tile(signal.dimshuffle(0,'x',1), (1,6,1)), 
        scene[:,word_orders]
    ].prod(-1)
        
    word_order_weights = theano_normalize(
        word_order_weights_original 
        + learning_weights[:,None]*word_order_to_add,
        1
    )
    
    return interpretation_weights, word_order_weights


def probs_languages_to_probs_scenes_multiple_participants(interpretation_weights, word_order_weights, 
                                    signals, scenes_trials, word_orders):
    """
    Finds probability of choosing each of the four scenes in `scenes`
    given the observed utterance and the current probabilities 
    of word-meaning association and word orders
    """
    
    # meaning that each word would be expressing
    # assuming each word order is true
    # dims (trial, participant, scene in trial, word orders, 3), values: meanings
    # e.g., (40, 5, 4, 6, 3)
    meanings_trials_orders = scenes_trials[:,:,:,word_orders]
    
    return normalize(
        (
            # get the probability of the words observed
            # in each trial expressing each scene
            # assuming each word order in turn
            interpretation_weights[
                tt.arange(interpretation_weights.shape[0])[:,None,None,None,None],
                tt.arange(interpretation_weights.shape[1])[None,:,None,None,None],
                signals[:,:,None,None,:],
                meanings_trials_orders
            ]
            # get joint probability of the observed words
            # (for each trial, word order and scene)
            .prod(-1) 
            # multiply by probability of word orders
            * word_order_weights.dimshuffle(0, 1, 'x', 2)
        # marginalize out word order
        # to get probs of observed words in scene
        ).sum(-1),
        2
    )


def factory_weight_model_multiple_participants(true_scenes_trials, signals, word_orders,
                                               history_choices_indices, scenes_trials,
                                               hierarchical_order_prior=True,
                                               hierarchical_learningweights=True,
                                               save_probs_order=False):
    """
    Parameters
    ----------
    history_choices_indices: None or array
        If none, the model is defined without observed
    """
    
    # with all these dimensions I've decided to be careful 
    # and explicitly write down all the shapes
    n_trials, n_participants, n_scenes, _ = scenes_trials.shape
    coords = {
        "trial": np.arange(n_trials),
        "participant": np.arange(n_participants),
        "scene": np.arange(n_scenes),
        "component": ['agent', 'action', 'scene'],
        "word_order": np.arange(6),
        "sentence_position": np.arange(3)
    }
    
    with pm.Model(coords=coords) as model:

        ##### Create data
        
        true_scenes_trials_data = pm.Data(
            'true_scenes_trials',
            true_scenes_trials,
            dims=('trial', 'participant', 'component')
        )

        signals_data = pm.Data(
            'signals',
            signals,
            dims=('trial', 'participant', 'sentence_position')
        )

        word_orders_data = pm.Data(
            'word_orders',
            word_orders,
            dims=('word_order', 'sentence_position')
        )

        scenes_trials_data = pm.Data(
            'scenes_trials',
            scenes_trials,
            dims=('trial', 'participant', 'scene', 'component')
        )
        
        if history_choices_indices is None:
            history_choices_indices_data = None
        else:
            history_choices_indices_data = pm.Data(
                'history_choices_indices',
                history_choices_indices,
                dims=('trial', 'participant')
            )

        ####### Define priors
                    
        if hierarchical_order_prior:
            
            # sample population-level hyperprior
            # over the individual parameters of the Dirichlet following this: 
            # http://tdunning.blogspot.com/2010/04/sampling-dirichlet-distribution-revised.html
            gammas = pm.Gamma(
                'hyper_gammas',
                alpha=5,
                beta=2,
                dims=('word_order')
            )

            hyper_alpha = pm.Deterministic(
                'hyper_alpha',
                gammas.sum()
            )

            # this is added just for record 
            # but not used
            pm.Deterministic(
                'hyper_ms',
                gammas / hyper_alpha,
                dims=('word_order')
            )

            # shape (# participants, 6)
            # dim (participant, word order)
            word_order_weights = pm.Dirichlet(
                'word_order_prior', 
                gammas,
                dims=('participant', 'word_order')
            )
        
        else:
            
            # shape (# participants, 6)
            # dim (participant, word order)
            word_order_weights = pm.Dirichlet(
                'word_order_prior', 
                [1]*6,
                dims=('participant', 'word_order')
            )
        
        
        if hierarchical_learningweights:
            
            learning_weights_mu = pm.HalfNormal(
                'learning_weights_mu',
                sigma=3
            )
            
            learning_weights_sigma = pm.HalfNormal(
                'learning_weights_sigma',
                sigma=3
            )
            
            # shape (# participants)
            learning_weights = pm.TruncatedNormal(
                'learning_weights',
                mu=learning_weights_mu,
                sigma=learning_weights_sigma,
                lower=0,
                dims=('participant')
            )
            
        else:
            
            # shape (# participants)
            learning_weights = pm.Gamma(
                'learning_weights',
                alpha=8.,
                beta=15.,
                dims=('participant')
            )
        
        ####### likelihood part
        
        # The participants' marginal prior over 
        # interpretation weights is fixed in advance
        # to be uniform, rather than estimated.
        # dims (participant, word, meaning)
        interpretation_weights = theano_normalize(
            tt.ones((n_participants,7,7)),
            axis=1
        )

        # scan is theano's way to do looping.
        # probs of each word-meaning connection
        # and each word order respectively
        (probs_interpretations, probs_word_orders), outputs_info = theano.scan(
            update_weights_theano_multiple_participants,
            sequences=[
                true_scenes_trials_data,
                signals_data
            ],
            outputs_info=[
                interpretation_weights,
                word_order_weights
            ],
            non_sequences=[
                word_orders_data,
                learning_weights
            ]
        )
        
        if save_probs_order:
            pm.Deterministic(
                'probs_orders',
                probs_word_orders
            )

        # dims (trial, participant, scenes in trial)
        prob_scenes = probs_languages_to_probs_scenes_multiple_participants(
            probs_interpretations, 
            probs_word_orders, 
            signals_data, 
            scenes_trials_data, 
            word_orders_data
        )

        pm.Categorical(
            'chosen_scenes',
            p=prob_scenes,
            observed=history_choices_indices_data,
            dims=('trial', 'participant')
        )
        
    return model


def prior_predictive_sample(n_trials, n_participants):
    
    simulated_results = simulate_full_experiment(
        n_trials, 
        n_participants,             
        true_languages_setting='unique', 
        simulate_responses=False
    )

    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )

    model = factory_weight_model_multiple_participants(
        true_scenes_trials=simulated_results['true_scenes_trials'], 
        signals=simulated_results['signals'], 
        scenes_trials=simulated_results['scenes_trials'],
        word_orders=word_orders,
        history_choices_indices=None,
        save_probs_order=True,
        hierarchical_order_prior=True,
        hierarchical_learningweights=False
    )

    with model:
        simulated_data = pm.sample_prior_predictive(samples=1)

    return simulated_results, simulated_data


def get_and_fit_data(participant_exclusion=True, method='hmc', first_n_trials=None,
                     fit_kwargs=None):
    """
    Parameters
    ----------
    participant_exclusion: Bool
        Whether participants are excluded
    method: string
        Either 'hmc' or 'variational'
    first_n_trials: None or int
        If int, the first_n_trials trials are considered
        If None, all trials are considered
    """
    
    data = get_data()

    analysis_arrays = get_analysis_arrays(
        data,
        exclude_nonimproving_participants=participant_exclusion
    )
    
    print('Got and wrangled the data')
    
    if first_n_trials is not None:
        print(f'Getting only the first {first_n_trials} trials')
        analysis_arrays = get_first_n_trials(
            analysis_arrays,
            n_trials=first_n_trials
        )

    _,_,_, word_orders = define_objects(
        full_output=True
    )

    model = factory_weight_model_multiple_participants(
        analysis_arrays['true_scenes_trials'],
        analysis_arrays['signals'],
        word_orders,
        analysis_arrays['history_choices_indices'],
        analysis_arrays['scenes_trials']
    )
    
    print('Built the model')
    
    helps = first_n_trials if first_n_trials is None else 'all'
    outputfile_name = (
        'results/'
        f'method-{method}'
        f'_excluded-{participant_exclusion}'
        f'_trialsupto-{helps}'
        '.cdf'
    )

    if method=='variational':
        try:
            trace = az.from_netcdf(
                outputfile_name
            )
        except FileNotFoundError:
            with model:
                if fit_kwargs is None:
                    fit_kwargs = {'n': 50000}
                fit = pm.fit(
                    **fit_kwargs
                )
            with model:
                trace = az.from_pymc3(
                    fit.sample(1000)
                )
            az.to_netcdf(
                trace, 
                outputfile_name
            )
            print('Saved samples in results folder')

    elif method=='hmc':
        try:
            trace = az.from_netcdf(
                outputfile_name
            )
        except FileNotFoundError:
            with model:
                if fit_kwargs is None:
                    fit_kwargs = {
                        'draws': 2000, 
                        'return_inferencedata': True
                    }
                trace = pm.sample(
                    **fit_kwargs
                )
            az.to_netcdf(
                trace, 
                outputfile_name
            )
            print('Saved samples in results folder')
    else:
        raise ValueError('Method not implemented! Choose hmc or variational')
        
    return data, analysis_arrays, model, trace
    
if __name__=='__main__':
    
    get_and_fit_data(
        participant_exclusion=True, 
        method='hmc',
        first_n_trials=20,
        fit_kwargs={
            'cores': 1, 
            'draws': 1000
        }
    )