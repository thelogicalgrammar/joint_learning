import numpy as np
import itertools as it

try:
    from IPython.display import display
    from ipywidgets import IntProgress
    displaybar = True
except ImportError:
    displaybar = False
    print("Couldn't import progress bar stuff. Fine if you're just fitting model")


def normalize(arr, axis=0):
    return arr / arr.sum(axis, keepdims=True)

def define_objects(n_beings=4, n_actions=3, full_output=False):
    
    # call beings with [0,1,2,3]
    being = range(n_beings)
    # call actions with [4,5,6]
    # NOTE: calling beings and actions 
    # with different numbers is useful below
    # when using them as indices of the language array
    actions = range(n_beings, n_beings+n_actions)
    signals = range(n_beings+n_actions)

    # columns: (agent, action, patient)
    # E.g. [0,0,1] means 'first being as agent, first action, second being as patient'
    scenes = np.array([
        i
        for i in it.product(being, actions, being)
        # Note: agent and patient can't be the same
        if i[0] != i[2]
    ])

    # Contains the possible interpretation 
    # functions, modelled as a function from
    # meanings to signals
    # For instance [2,1,3,...]
    # Means that meaning 0 has signal 2
    # meaning 1 has signal 1 etc.
    # Shape: (function, signal index)
    language_interpret = np.array(list(it.permutations(signals)))

    # Possible word orders 
    # 0: S, 1: V, 2: O
    word_orders = np.array(list(it.permutations(range(3))))

    # For every combination of interpretation function, 
    # word order, and scene, I need to produce which three words 
    # would be used by the language for that scene.

    # Contains for each language and scene
    # The index of the three signals that 
    # describe the scene in that language
    # dims: (interpretation function, word order, scene, 3)
    languages = []
    # for each interpretation function
    for inter in language_interpret:
        sub = []
        # for each word order
        for order in word_orders:
            subsub = []
            # for each scene
            # scene has: (agent, action, patient)
            for scene in scenes:
                # signals for [agent, action, patient]
                subsub.append(inter[scene][order])
            sub.append(subsub)
        languages.append(sub)
    languages = np.array(languages)
    
    if full_output:
        return scenes, languages, language_interpret, np.array(word_orders)
    return scenes, languages


def generate_trials(lang, n, scenes, n_scenes_trial):
    """
    Generate n trials from lang.
    Parameters
    ----------
    lang: array
        An array with shape (36,3)
        Containing for each scene 
        the three words used for that scene
    n: int
        The number of trials to return
    scenes: array
        The array containing the scenes
        see above for description of array
    n_scenes_trial: int
        The total number of scenes shown in each trial
        (one is true and others are confounders)
    Returns
    -------
    list of arrays
        Each array has len (trials)
        (true_scene_index, four possible choices, utterance)
    """
    index_scenes = np.array([
        np.random.choice(
            len(scenes), 
            4,
            replace=False
        )
        for _ in range(n)
    ])
    
    true_scenes_index = np.random.choice(
        np.arange(n_scenes_trial), 
        size=len(index_scenes)
    )
    true_scenes = index_scenes[
        np.arange(len(index_scenes)),
        true_scenes_index
    ]
    utterances = lang[true_scenes]
    
    return true_scenes, index_scenes, utterances


def define_priors(word_orders, language_interpret):
    word_order_prior = normalize(
        np.array([1]*len(word_orders))
    ) # normalize([1,1,1,1,1,5])
    interpret_prior = normalize(
        np.array([1]*len(language_interpret))
    )
    return word_order_prior, interpret_prior


###### The following functions run the perfect Bayesian learner, 
# but they are here because they're also used below to simulate experiments:

def choose_scene(utterance_by_language, indices_scenes, current_state, return_p_scene=False):
    """
    Choose a scene based on current posterior
    by sampling with model averaging.
    The main step is to restrict to the 4 available scenes
    and to the given utterance.
    Get for each scene P(scene | utterance, language)P(language)
    And then sum across languages
    """
    # Joint probability of each scene & language
    # given the utterance.
    # restrict to just available scenes
    p_lang_scene_given_utt = utterance_by_language[:,:,indices_scenes] 
    # multiply with prior over languages
    p_lang_scene_given_utt = p_lang_scene_given_utt * current_state[:,:,None]
    # sum over languages to get probability of 
    # each of the four scenes given the utterance
    p_scene = normalize(p_lang_scene_given_utt, axis=(0,1,2)).sum((0,1))
    chosen_scene = np.random.choice(indices_scenes, p=p_scene)
    if return_p_scene:
        return chosen_scene, p_scene
    return chosen_scene


def update_knowledge(chosen_scene, true_scene, languages, utterance, 
                     indices_scenes, current_state, negative_evidence, 
                     noise=0, noise_type='uniform'):
    """
    Update current knowledge with Bayesian update
    """
    
    if chosen_scene == true_scene:
        # update current_state to only keep those languages
        # that are compatible with that scene and utterance
        compatible_languages = (languages == utterance).all(-1)[:,:,chosen_scene]

    elif negative_evidence:
        # otherwise it's one of the remaining three 
        # observed scenes.
        # Get indices of non chosen scenes
        indices_non_chosen = indices_scenes[indices_scenes!=chosen_scene]
        # Mask for languages that use the seen utterance
        # for any of the non-chosen scenes
        compatible_languages = (languages == utterance).all(-1)[:,:,indices_non_chosen].any(-1)
    else:
        # if no negative evidence, all languages are compatible
        # this is effectively the likelihood function
        compatible_languages = np.ones_like(current_state)

    # this is 1 with a probability = noise
    # when it is 1, it corresponds to a language that is 
    # incorrectly not considered impossible
    random_bool = np.random.rand(*compatible_languages.shape) < noise

    if noise_type == 'ignore':
        likelihood = np.where(
            compatible_languages,
            1,
            random_bool
        )
    elif noise_type == 'uniform':
        likelihood = np.where(
            compatible_languages,
            1-noise,
            noise
        )
    else:
        raise ValueError(f'Noise type {noise_type} not recognized!')
    
    current_state = normalize(likelihood*current_state, axis=(0,1))
    return current_state
    

def run_experiment(trials, interpret_prior, word_order_prior, 
                   negative_evidence, noise, return_p_scene=False, 
                   noise_type='uniform'):
    
    scenes, languages = define_objects()
    
    # dims: (interpretation function, word order)
    # Contains the probabilities of each language,
    # encoded as a combination of intepretation function 
    # and word order
    current_state = interpret_prior[:,None] * word_order_prior

    # loop through the observations and update the posterior
    # at each timestep
    history_states = [current_state]
    history_choices = []
    p_scenes = []
    for true_scene, indices_scenes, utterance in zip(*trials):

        # get indicator of utterance for each language
        # NOTE: some languages don't use a certain
        # utterance at all!
        # This gives P(scenes | language, utterance)
        # Which is always 0 or 1
        utterance_by_language = (languages == utterance).all(-1)

        chosen_scene, p_scene = choose_scene(
            utterance_by_language, 
            indices_scenes, 
            current_state,
            return_p_scene=True
        )
        p_scenes.append(p_scene)
        history_choices.append(chosen_scene)

        current_state = update_knowledge(
            chosen_scene, 
            true_scene, 
            languages, 
            utterance, 
            indices_scenes,
            current_state,
            negative_evidence,
            noise,
            noise_type
        )
        history_states.append(current_state)
        
    if return_p_scene:
        return history_states, history_choices, p_scenes
    return history_states, history_choices


def run_experiments(index_true_lang, interpret_prior, word_order_prior, n, 
                    negative_evidence, return_p_scene=False):
    
    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )
    
    word_order_prior, interpret_prior = define_priors(
        word_orders, language_interpret
    )
    
    if displaybar:
        f = IntProgress(min=0, max=n)
        display(f)

    histories = []
    for i in range(n):
        index_lang = (
            index_true_lang 
            if type(index_true_lang) is tuple
            else index_true_lang[i]
        )
        trials = generate_trials(
            languages[index_lang], 
            10, 
            scenes, 
            4
        )
        histories.append(
            run_experiment(
                trials, 
                interpret_prior, 
                word_order_prior,
                negative_evidence,
                return_p_scene=return_p_scene
            )[0]
        )
        if displaybar:
            f.value += 1
    return histories


def simulate_full_experiment(n_trials, n_participants, true_languages_setting,
                             simulate_responses, negative_evidence=False,
                             noise=0, return_p_scene=False, noise_type='uniform'):
    """
    Simulate data from multiple perfeclty bayesian participants
    
    Parameters
    ----------
    simulate_responses: bool
        Whether the dependent variable is simulated
    negative_evidence: bool
        Whether the participant updates their knowledge
        when they get the wrong scene (based on the true scene
        which they see, as in the experiment)
    noise: float
        The noise of the Bayesian update
        (0 means no noise, 1 means full noise)
    noise_type: str
        The type of noise to add
        'uniform': likelihood function is smoothed out
        'ignore': a language that is excluded by the data is 
            nonetheless kept to 1 with probability "noise"
    """

    if return_p_scene:
        assert simulate_responses, "return_p_scene is only available if simulate_responses is True"

    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )

    word_order_prior, interpret_prior = define_priors(
        word_orders, 
        language_interpret
    )

    true_scenes = []
    indices_scenes_trials = []
    signals = []
    
    if simulate_responses:
        history_states = []
        history_choices = []
        history_choices_indices = []

    if displaybar:
        # in case progressbar was not defined
        f = IntProgress(min=0, max=n_participants)
        display(f)

    if true_languages_setting == 'half':
        # pick a random true language,
        # represented as a tuple of
        # (interpretationfunction, word_order)
        indices_true_language = (
            [
                (
                    np.random.randint(low=0, high=len(languages)), 
                    np.random.randint(low=0, high=6)
                )
            ]*(n_participants//2) +
            [
                (
                    np.random.randint(low=0, high=len(languages)), 
                    np.random.randint(low=0, high=6)
                )
            ]*(n_participants//2)
        )
        
    elif true_languages_setting == 'unique':
        ### Train each participant on different lang
        indices_true_language = [
            (
                np.random.randint(low=0, high=len(languages)), 
                np.random.randint(low=0, high=6)
            ) 
            for _ in range(n_participants)
        ]
        
    elif true_languages_setting == 'same':
        ### Train all on the same language
        indices_true_language = [
            (
                np.random.randint(low=0, high=len(languages)), 
                np.random.randint(low=0, high=6)
            ) 
        ] * n_participants
        
    else:
        raise ValueError('Setting unknown!')

    if return_p_scene:
        history_p_scenes = []

    for i in range(n_participants):

        if isinstance(noise, int) or isinstance(noise, float):
            noise_i = noise
        else:
            noise_i = noise[i]
        
        index_true_language = indices_true_language[i]

        trials = generate_trials(
            languages[index_true_language], 
            n=n_trials, 
            scenes=scenes, 
            n_scenes_trial=4
        )

        t_scenes, i_scenes_trials, sign = trials

        true_scenes.append(t_scenes)
        indices_scenes_trials.append(i_scenes_trials)
        signals.append(sign)

        if simulate_responses:
            
            # Run perfect Bayesian learner here
            output = run_experiment(
                trials, 
                interpret_prior, 
                word_order_prior, 
                negative_evidence=negative_evidence,
                noise=noise_i,
                return_p_scene=return_p_scene,
                noise_type=noise_type
            )

            if return_p_scene:
                h_states, h_choices, p_scenes = output
            else:
                h_states, h_choices = output

            history_states.append(h_states)
            history_choices.append(h_choices)
            if return_p_scene:
                history_p_scenes.append(p_scenes)

            # go from the actual chosen scenes to their indices
            history_choices_indices.append(
                np.argwhere(
                    i_scenes_trials == np.array(h_choices)[:,None]
                )[:,1]
            )
            
        if displaybar:
            f.value += 1

    # transform into numpy arrays AND reshuffle 
    # so trial is first dimension (which is needed below when using `scan`)
    true_scenes = np.swapaxes(true_scenes, 0, 1)
    indices_scenes_trials = np.swapaxes(indices_scenes_trials, 0, 1)
    signals = np.swapaxes(signals, 0, 1)
    if return_p_scene:
        history_p_scenes = np.swapaxes(history_p_scenes, 0, 1)

    scenes_trials = scenes[indices_scenes_trials]
    true_scenes_trials = scenes[true_scenes]
    
    returndict = {
        'true_scenes': true_scenes, 
        'indices_scenes_trials': indices_scenes_trials, 
        'signals': signals, 
        'scenes_trials': scenes_trials, 
        'true_scenes_trials': true_scenes_trials,
        'indices_true_language': np.array(indices_true_language)
    }
    
    if simulate_responses:
        
        history_states = np.swapaxes(history_states, 0, 1)
        history_choices = np.swapaxes(history_choices, 0, 1)
        history_choices_indices = np.swapaxes(history_choices_indices, 0, 1)
        
        returndict.update({
            'history_states': history_states, 
            'history_choices': history_choices, 
            'history_choices_indices': history_choices_indices
        })

    if return_p_scene:
        returndict.update({
            'history_p_scenes': history_p_scenes
        })
    
    return returndict
