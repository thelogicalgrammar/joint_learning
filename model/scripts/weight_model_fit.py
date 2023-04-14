import numpy as np
import pandas as pd
import arviz as az
import pickle

from .simulation_functions import define_objects, normalize, simulate_full_experiment, define_objects
from .data_functions import get_data, get_first_n_trials, get_analysis_arrays


import subprocess
import sys

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [
    r.decode().split('==')[0] 
    for r in reqs.split()
]

if 'pymc3' in installed_packages:
    print("Pymc3 found")
    import pymc3 as pm
    import theano
    import theano.tensor as tt
elif 'aesara' in installed_packages:
    print("Aesara found, import pymc v4")
    import pymc as pm
    import aesara as theano
    import aesara.tensor as tt
elif 'pytensor' in installed_packages:
    print("Pymc v4 not found, trying pymc v5")
    import pymc as pm
    import pytensor as theano
    import pytensor.tensor as tt
else:
    raise ImportError("No pymc3, pymc4 or pymc5 found")

tprint = theano.printing.Print

print("Beginning: Using this version of pymc: ", pm.__version__)

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
                tt.shared(scenes[true_scenes]),
                tt.shared(signals)
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

    # az.plot_trace(trace, compact=False)
    # plt.tight_layout()
    # plt.show()

    with model:
        fit = pm.fit()

    with model:
        fit_samples = az.from_pymc3(fit.sample(5000))
    # az.plot_trace(fit_samples, compact=False)
    # plt.tight_layout()
    # plt.show()
    
    
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
    learning_weights: array
        dims: (participant)
        The learning weight of each participant
    Returns
    -------
    tuple of arrays
        interpretation_weights: array
            Same shape as interpretation_weights_original
            NOTE: normalized before returning
        word_order_weights: array
            Same shape as word_order_weights_original
            NOTE: normalized before returning
    """
    
    n_parti = interpretation_weights_original.shape[0]
    
    ### Snippet A (update interpretation weights)

    # get one-hot vectors for the indices as described in the notebook
    # in which to add the calculated weights that should be added.
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
        # to get the probability that it's ANY of those orders
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
        # create an array with shape (participant, signal, meaning)
        tt.zeros(interpretation_weights_original.shape)[
            arg1, arg2, arg3
        ],
        values_to_add
    )
    
    # Create array with new weights and then normalize
    # in the meaning dimension 
    # (get a distribution over meanings given a signal
    # which makes sense since this is an interpretation matrix)
    # shape: participant, signal, meaning
    interpretation_weights = theano_normalize(
        # add to the original array
        interpretation_weights_original 
        + learning_weights[:,None,None]*interpretation_to_add,
        axis=-1
    )
    
    # Snippet B (update word order weights)
    
    # weights to add to the word order
    word_order_to_add = interpretation_weights_original[
        tt.tile(
            tt.arange(n_parti).dimshuffle(0,'x','x'), 
            (1,6,3)
        ),
        # for each participant, 
        # repeat signal once for each word order.
        # dims (participant, word order, signal)
        tt.tile(
            signal.dimshuffle(0,'x',1), 
            (1,6,1)
        ), 
        scene[:,word_orders]
    ].prod(-1)
        
    word_order_weights = theano_normalize(
        word_order_weights_original 
        + learning_weights[:,None]*word_order_to_add,
        1
    )
    
    return interpretation_weights, word_order_weights


def softmax_theano(x, axis):
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def probs_languages_to_probs_scenes_multiple_participants(
                    interpretation_weights, word_order_weights, 
                    signals, scenes_trials, word_orders, softmax_alphas):
    """
    Finds probability of choosing each of the four scenes in `scenes`
    given the observed utterance and the current probabilities 
    of word-meaning association and word orders
    
    Parameters
    ----------
    interpretation_weights: array
        Dimensions: (trial, participant, word, meaning)
        NOTE: Trial dimension was added in the scan
    word_order_weights: array
        Dimensions: (trial, participant, word order)
        NOTE: Trial dimension was added in the scan
    signals: array
        Dimensions: (trial, participant, sentence position)
    scenes_trials: array
        Dimensions: (trial, participant, scene, component)
    word_orders: array
        Dimensions: (word order, sentence position)
    """
    
    # meaning that each word would be expressing
    # assuming each word order is true
    # dims (trial, participant, scene in trial, word orders, 3), values: meanings
    # e.g., (40, 5, 4, 6, 3)
    meanings_trials_orders = scenes_trials[:,:,:,word_orders]
    
    # get the probability that the whole sentence 
    # observed in each trial expresses 
    # each of the scenes in the trial
    # assuming each word order in turn
    # Dimensions: 
    # (trial, participant, scene in trial, word order)
    prob_sentence_by_trial_given_order = (
        interpretation_weights[
            tt.arange(interpretation_weights.shape[0])[:,None,None,None,None],
            tt.arange(interpretation_weights.shape[1])[None,:,None,None,None],
            signals[:,:,None,None,:],
            meanings_trials_orders
        ]
        # get joint probability of the observed words
        # (for each trial, word order and scene)
        .prod(-1) 
    )
    
    # Marginalize out word order
    # to get probs of observed words in scene
    # Dimensions: (trial, participant, scene in trial)
    unnormalized_probs = (
        prob_sentence_by_trial_given_order
        # multiply by probability of word orders
        * word_order_weights.dimshuffle(0, 1, 'x', 2)
    ).sum(-1)
    
    if softmax_alphas is None:
        prodprobs = normalize(
            unnormalized_probs,
            2
        )
        
    else:
        prodprobs = softmax_theano(
            unnormalized_probs*softmax_alphas[:,None],
            -1
        )
        
    return prodprobs


def factory_weight_model_multiple_participants(true_scenes_trials, signals, word_orders,
                                               history_choices_indices, scenes_trials,
                                               hierarchical_order_prior=True,
                                               hierarchicallearningweights=False,
                                               save_probs_order=False,
                                               softmax_choice=True,
                                               store_p_correct=False):
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
        
        
        if hierarchicallearningweights:
            
            
#             This commented definition of the
#             hierarchical structure causes an error:

#             learning_weights_mu = pm.HalfNormal(
#                 'learning_weights_mu',
#                 sigma=2
#             )
            
#             learning_weights_sigma = pm.HalfNormal(
#                 'learning_weights_sigma',
#                 sigma=2
#             )
            
            zs_sigma = pm.Normal(
                'zetas_sigma',
                dims=('participant')
            )
            
            learning_weights_mu = pm.Normal(
                'learning_weights_mu',
                mu=0.,
                sigma=0.5
            )
        
            learning_weights_sigma = pm.HalfNormal(
                'learning_weights_sigma',
                sigma=0.1
            )
            
            # shape (# participants)
            learning_weights = pm.Deterministic(
                'learning_weights',
                pm.math.exp(learning_weights_mu + zs_sigma * learning_weights_sigma),
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
            
        if softmax_choice:
            softmax_alphas = pm.Exponential(
                'softmax_alpha',
                lam=0.5,
                dims=('participant')
            )
        else:
            softmax_alphas = None
        
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
        # where there are always 4 scenes in the trial
        prob_scenes = probs_languages_to_probs_scenes_multiple_participants(
            probs_interpretations, 
            probs_word_orders, 
            signals_data, 
            scenes_trials_data, 
            word_orders_data,
            softmax_alphas
        )
        
        # scene 0 is always the correct one in 
        # the experiment's dataset
        if store_p_correct:
            pm.Deterministic(
                'p_correct',
                prob_scenes[:,:,0]
            )

        pm.Categorical(
            'chosen_scenes',
            p=prob_scenes,
            observed=history_choices_indices_data,
            dims=('trial', 'participant')
        )
        
    return model


def prior_predictive_sample(n_trials, n_participants, factory_kwargs=None):
    """
    Take prior predictive samples, i.e., run simulated experiment
    """
    
    if factory_kwargs is None:
        factory_kwargs = {
            'save_probs_order': False,
            'hierarchical_order_prior': True,
            'hierarchicallearningweights': True,
            'softmax_choice': True
        }

    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )
    
    simulated_results = simulate_full_experiment(
        n_trials, 
        n_participants,
        # unique means each participant gets their own true language
        true_languages_setting='unique', 
        simulate_responses=False
    )

    model = factory_weight_model_multiple_participants(
        true_scenes_trials=simulated_results['true_scenes_trials'], 
        signals=simulated_results['signals'], 
        scenes_trials=simulated_results['scenes_trials'],
        word_orders=word_orders,
        history_choices_indices=None,
        **factory_kwargs
    )

    with model:
        simulated_data = pm.sample_prior_predictive(samples=1)
    
    return simulated_results, simulated_data


def simulate_parameter_recovery(n_trials=150, n_participants=150, 
                                recovery_method='variational', n=0,
                                save_path='param_recovery/'):
    """
    Put this in a function so I can run it as a script on the server.
    """
    
    factory_kwargs = {
        'save_probs_order': False,
        'hierarchical_order_prior': True,
        'hierarchicallearningweights': True,
        'softmax_choice': True
    }
        
    simulated_results, simulated_data = prior_predictive_sample(
        n_trials, 
        n_participants,
        factory_kwargs=factory_kwargs
    )
    
    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )
    
    ppc_model = factory_weight_model_multiple_participants(
        simulated_results['true_scenes_trials'],
        simulated_results['signals'],
        word_orders,
        simulated_data['chosen_scenes'][0],
        simulated_results['scenes_trials'],
        **factory_kwargs
    )
    
    output = {
        'results': simulated_results,
        'data': simulated_data
    }
    
    if recovery_method=='variational':

        print("Running variational fit")
    
        with ppc_model:
            ppc_fit = pm.fit(n=50000)

        with ppc_model:
            ppc_fit_samples = az.from_pymc3(
                ppc_fit.sample(1000)
            )
            
        output['samples'] = ppc_fit_samples

        with open(f'{save_path}ppc_samples_variational_{n}.pickle', 'wb') as openfile:
            pickle.dump(
                output, 
                openfile
            )
    
    elif recovery_method=='hmc':

        print("Running hmc fit")
        
        with ppc_model:
            ppc_samples = pm.sample(
                return_inferencedata=True,
                tune=1000,
                draws=1000,
            )
        
        output['samples'] = ppc_samples
            
        with open(f'{save_path}ppc_samples_hmc_{n}.pickle', 'wb') as openfile:
            pickle.dump(
                output, 
                openfile
            )


def get_and_fit_data(participant_exclusion=True,
                     method='hmc', first_n_trials='all',
                     fit_kwargs=None, model_kwargs=None, save=True, datapath=None,
                     outputfile_append='', save_path='results/'):
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
    
    data = get_data(
        datapath
    )

    analysis_arrays = get_analysis_arrays(
        data,
        exclude_nonimproving_participants=participant_exclusion
    )
    
    print('Got and wrangled the data')

    if first_n_trials is not None:
        print(f'Getting only the first {first_n_trials} trials')
        analysis_arrays = get_first_n_trials(
            analysis_arrays,
            n_trials= first_n_trials
        )

    _,_,_, word_orders = define_objects(
        full_output=True
    )

    if model_kwargs is None:
        model_kwargs = dict()
    
    model = factory_weight_model_multiple_participants(
        analysis_arrays['true_scenes_trials'],
        analysis_arrays['signals'],
        word_orders,
        analysis_arrays['history_choices_indices'],
        analysis_arrays['scenes_trials'],
        **model_kwargs
    )
    
    print('Built the model')
    
    added_fit = (
        '-' 
        '_'.join(f'{v}-{k}' for v,k in fit_kwargs.items())
    ) if fit_kwargs is not None else ''
    
    outputfile_name = save_path+(
        f'method-{method}'
        f'_excluded-{participant_exclusion}'
        f'_trialsupto-{first_n_trials}'
        +added_fit
        +outputfile_append
        +'.cdf'
    )
    
    print(f"Looking at file with name {outputfile_name}")

    try:
        print('Checking test point: ')
        print(model.check_test_point())
    except AttributeError:
        print('check_test_point is not defined')

    if method=='variational':
        try:
            trace = az.from_netcdf(
                outputfile_name
            )
            print("Already found a file with that name, got from file")
        except FileNotFoundError:
            if fit_kwargs is None:
                fit_kwargs = {
                    'n': 50000
                }

            with model:

                # advi = pm.ADVI()

                # tracker = pm.callbacks.Tracker(
                #     mean=advi.approx.mean.eval, 
                #     std=advi.approx.std.eval,  
                # )

                # fit = advi.fit(
                #     callbacks=[tracker],
                #     **fit_kwargs
                # )

                fit = pm.fit(
                    **fit_kwargs
                )

            # print("Mean: ", tracker['mean'])
            # print("Std: ", tracker['std'])

#             fig = plt.figure(figsize=(16, 9))
#             mu_ax = fig.add_subplot(221)
#             std_ax = fig.add_subplot(222)
#             hist_ax = fig.add_subplot(212)
#             mu_ax.plot(tracker["mean"])
#             mu_ax.set_title("Mean track")
#             std_ax.plot(tracker["std"])
#             std_ax.set_title("Std track")
#             hist_ax.plot(advi.hist)
#             hist_ax.set_title("Negative ELBO track");
#             plt.show()

            with model:
                pmtrace = fit.sample(
                    1000
                )
                trace = az.from_pymc3(
                    pmtrace
                )

            if save:
                try:
                    az.to_netcdf(
                        trace, 
                        outputfile_name
                    )
                    print('Saved samples in results folder')
                except PermissionError:
                    az.to_netcdf(
                        trace, 
                        'model/'+outputfile_name
                    )
                    print('Saved samples in results folder')

    elif method=='hmc':
        try:
            trace = az.from_netcdf(
                outputfile_name
            )
            print("Already found a file with that name, got from file")
            
        except FileNotFoundError:
            with model:
                if fit_kwargs is None:
                    fit_kwargs = {
                        'draws': 2000, 
                        'return_inferencedata':True
                    }
                trace = pm.sample(
                    **fit_kwargs,
                )
            if save:
                az.to_netcdf(
                    trace, 
                    outputfile_name
                )
                print('Saved samples in results folder')

    elif method=='jax':
        
        ##### TODO! Does not work yet!
        
        print('Using jax')

        # Override imports above since we're gonna need
        # pymc v4 if we use JAX
        import pymc as pm
        import aesara as theano
        import aesara.tensor as tt
        tprint = theano.printing.Print

        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        import pymc.sampling_jax
        from aesara.link.jax.dispatch import jax_funcify


        try:
            trace = az.from_netcdf(
                outputfile_name
            )
            print("Already found a file with that name, got from file")
            
        except FileNotFoundError:
            with model:
                if fit_kwargs is None:
                    fit_kwargs = {
                        'draws': 2000, 
                    }
                trace = sampling_jax.sample_numpyro_nuts(
                    **fit_kwargs
                )
            if save:
                az.to_netcdf(
                    trace, 
                    outputfile_name
                )
                print('Saved samples in results folder')

    elif method=='map':
        try:
            trace = az.from_netcdf(
                outputfile_name
            )
            print("Already found a file with that name, got from file")
            
        except FileNotFoundError:
            with model:
                # technically not a trace
                trace = pm.find_MAP()
            if save:
                try:
                    az.to_netcdf(
                        trace, 
                        outputfile_name
                    )
                except PermissionError:
                    az.to_netcdf(
                        trace, 
                        'model/'+outputfile_name
                    )
                print('Saved MAP in results folder')
    else:
        raise ValueError('Method not implemented!')
        
    return data, analysis_arrays, model, trace


if __name__=='__main__':
    
    # get_and_fit_data(
    #     participant_exclusion=True, 
    #     method='hmc',
    #     # first_n_trials=100,
    #     fit_kwargs={
    #         'draws': 10,
    #         'tune': 10,
    #         'chains': 4
    #     },
    #     model_kwargs={
    #         'hierarchicallearningweights': False,
    #         'softmax_choice': True
    #     },
    #     datapath="../data.csv"
    # )

    get_and_fit_data(
        participant_exclusion=True, 
        method='hmc',
        first_n_trials=100,
        fit_kwargs={
            'draws': 10,
            'tune': 10,
            'chains': 4
        },
        model_kwargs={
            'hierarchicallearningweights': False,
            'softmax_choice': True
        },
        # datapath="../michael_data/results_2021-08-23T12_58_44_175Z_langlearning-v2.csv"
	datapath="../../data.csv"
    )
    
    # get_and_fit_data(
    #     participant_exclusion=True, 
    #     method='variational',
    #     model_kwargs={
    #         'hierarchicallearningweights': True,
    #         'softmax_choice': True,
    #         'store_p_correct': True
    #     },
    #     # datapath=
    # )
