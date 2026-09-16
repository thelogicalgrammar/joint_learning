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
    Generate n trials with the structured 2x2 foil design used in the
    Aclapa experiment. For a target (a, x, p), the four scenes are:

        target          (a,  x,  p)
        action foil     (a,  x', p)   — same entities, different action
        entity foil     (a', x,  p')  — same action, different entities
        a&e foil        (a', x', p')  — entity foil's entities + action
                                        foil's action

    x' is sampled uniformly from actions != x, and (a', p') is sampled
    uniformly from entity pairs != (a, p) with a' != p'.

    Parameters
    ----------
    lang: array, shape (n_scenes, 3)
        For each scene, the three signals the language uses for it.
    n: int
        Number of trials.
    scenes: array, shape (n_scenes, 3)
        Each row is (agent, action, patient) signal indices.
    n_scenes_trial: int
        Must be 4 (the structured design always yields 4 candidates).

    Returns
    -------
    true_scenes: array, shape (n,)
    index_scenes: array, shape (n, 4) — columns are
        [target, action_foil, entity_foil, ae_foil].
    utterances: array, shape (n, 3)
    """
    assert n_scenes_trial == 4, (
        "Structured foil design always produces 4 candidates per trial"
    )

    scene_lookup = {tuple(int(v) for v in s): i for i, s in enumerate(scenes)}
    beings = np.unique(np.concatenate([scenes[:, 0], scenes[:, 2]]))
    actions = np.unique(scenes[:, 1])

    true_scenes = np.empty(n, dtype=int)
    index_scenes = np.empty((n, 4), dtype=int)
    utterances = np.empty((n, lang.shape[1]), dtype=lang.dtype)

    for i in range(n):
        target_idx = np.random.randint(len(scenes))
        a, x, p = (int(v) for v in scenes[target_idx])

        action_alts = actions[actions != x]
        x_prime = int(np.random.choice(action_alts))

        entity_alts = [
            (int(ap), int(pp))
            for ap in beings for pp in beings
            if ap != pp and (int(ap), int(pp)) != (a, p)
        ]
        a_prime, p_prime = entity_alts[np.random.randint(len(entity_alts))]

        true_scenes[i] = scene_lookup[(a, x, p)]
        index_scenes[i] = [
            scene_lookup[(a, x, p)],
            scene_lookup[(a, x_prime, p)],
            scene_lookup[(a_prime, x, p_prime)],
            scene_lookup[(a_prime, x_prime, p_prime)],
        ]
        utterances[i] = lang[true_scenes[i]]

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
    total = p_lang_scene_given_utt.sum()
    if total == 0:
        # No live language explains the utterance for any candidate. The
        # caller (run_experiment) normally pre-empts this by passing a
        # rollback state; this branch is a last-resort uniform pick to
        # keep the simulation finite if even the rollback can't explain.
        p_scene = np.ones(len(indices_scenes)) / len(indices_scenes)
    else:
        p_scene = p_lang_scene_given_utt.sum((0, 1)) / total
    chosen_scene = np.random.choice(indices_scenes, p=p_scene)
    if return_p_scene:
        return chosen_scene, p_scene
    return chosen_scene


def update_knowledge(chosen_scene, true_scene, languages, utterance,
                     indices_scenes, current_state, negative_evidence,
                     noise=0, noise_type='uniform', feedback_on_false_choices=True,
                     n_particles=None, prior=None, sampled_indices=None):
    """
    Update current knowledge with Bayesian update
    """

    assert not (feedback_on_false_choices and negative_evidence), "feedback_on_false_choices and negative_evidence cannot both be True"

    # if the chosen scene is the true scene
    if chosen_scene == true_scene:
        # update current_state to only keep those languages
        # that are compatible with that scene and utterance
        compatible_languages = (languages == utterance).all(-1)[:,:,true_scene]

    # the true scene is revealed to the participant after the choice
    elif feedback_on_false_choices:
        # update current_state to only keep those languages
        # that are compatible with the true scene and utterance
        compatible_languages = (languages == utterance).all(-1)[:,:,true_scene]

    # the true scene is not revealed to the participant after the choice
    # but the participant is told that the chosen scene is not the true scene
    elif negative_evidence:
        # otherwise it's one of the remaining three 
        # observed scenes (the chosen scene is not the true scene)
        # Get indices of non chosen scenes
        indices_non_chosen = indices_scenes[indices_scenes!=chosen_scene]
        # Mask for languages that use the seen utterance
        # for any of the non-chosen scenes
        compatible_languages = (languages == utterance).all(-1)[:,:,indices_non_chosen].any(-1)
    else:
        # if no negative evidence and no feedback on false choices, all languages are compatible
        # this is effectively the likelihood function
        compatible_languages = np.ones_like(current_state)

    if noise_type == 'ignore':
        # this is 1 with a probability = noise
        # when it is 1, it corresponds to a language that is
        # incorrectly not considered impossible
        random_bool = np.random.rand(*compatible_languages.shape) < noise
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
    elif noise_type in ('particles', 'particles_negbin', 'particles_local'):
        # Limited-attention learner: each trial, sample N hypotheses
        # from current_state and *test* them. Any sampled particle whose
        # language is incompatible with the data gets killed (zeroed in
        # current_state); unsampled languages keep their weight untouched.
        # 'particles':         N = n_particles (deterministic).
        # 'particles_negbin':  N ~ NegBin(mean, dispersion) per trial,
        #                      sampled as a gamma-Poisson mixture so
        #                      mean and variance can be set independently
        #                      (variance = mean + mean^2/dispersion).
        #                      n_particles is a (mean, dispersion) pair.
        #                      N=0 means no update this trial.
        # 'particles_local':   the caller has already sampled particles
        #                      (and used them for choose_scene); we just
        #                      apply the kill step on the supplied
        #                      sampled_indices.
        assert n_particles is not None, "n_particles must be provided when noise_type='particles[_negbin/_local]'"
        assert noise is None, "noise must be None when noise_type='particles[_negbin/_local]' (use n_particles instead)"

        if noise_type == 'particles_local':
            if sampled_indices is None:
                # Caller couldn't sample (no live support inside the prior);
                # nothing to test, nothing to kill, state unchanged.
                return current_state
            sampled_flat = sampled_indices
        else:
            if noise_type == 'particles_negbin':
                mean, dispersion = n_particles
                rate = np.random.gamma(shape=dispersion, scale=mean / dispersion)
                n_particles_trial = int(np.random.poisson(rate))
                if n_particles_trial == 0:
                    return current_state
            else:
                n_particles_trial = int(n_particles)

            flat_state = current_state.flatten()
            if prior is not None:
                # Sample biased by the prior, restricted to currently-live
                # hypotheses (those that haven't been killed by a past
                # contradiction). Killing still operates on current_state
                # below — the prior only shapes which live hypotheses get
                # picked as particles.
                live_mask = flat_state > 0
                sampling_weights = prior.flatten() * live_mask
            else:
                sampling_weights = flat_state
            sampling_weights = sampling_weights / sampling_weights.sum()
            sampled_flat = np.random.choice(
                flat_state.size, size=n_particles_trial, p=sampling_weights, replace=True
            )

        unique_sampled = np.unique(sampled_flat)
        flat_compat = compatible_languages.flatten()
        indices_to_kill = unique_sampled[~flat_compat[unique_sampled].astype(bool)]

        flat_state = current_state.flatten()
        new_flat = flat_state.copy()
        new_flat[indices_to_kill] = 0
        total = new_flat.sum()
        if total == 0:
            # All currently-alive languages were sampled and incompatible
            # (rare); keep state unchanged.
            return current_state
        return (new_flat / total).reshape(current_state.shape)
    else:
        raise ValueError(f'Noise type {noise_type} not recognized!')

    current_state = normalize(likelihood*current_state, axis=(0,1))
    return current_state
    

def precompute_eval_arrays(eval_language, scenes, languages, n_eval=1000):
    """
    Sample n_eval hypothetical trials from eval_language and precompute
    per-language arrays so E[P(correct)] under any posterior can be
    estimated with two BLAS-accelerated matrix-vector products.

    A[i, interp, order] = 1.0 iff that language produces trial i's
        utterance for trial i's true scene, else 0.0.
    B[i, interp, order] = number of candidate scenes in trial i for
        which that language produces trial i's utterance.

    Stored as float32 (not int8) so that the matmul in
    estimate_p_correct dispatches to BLAS; numpy doesn't BLAS-accelerate
    integer matmuls and that path is ~10x slower than the float one.
    """
    true_scenes, indices_scenes, utterances = generate_trials(
        eval_language, n_eval, scenes, 4
    )
    n_interp, n_order = languages.shape[:2]

    # Encode each 3-signal utterance as a single int32. Touching 1/3 the
    # data per comparison gives ~50x speedup over the (lang == utt).all(-1)
    # form that materializes a 4D bool tensor per iteration.
    weights = np.array([49, 7, 1], dtype=np.int32)
    languages_enc = languages.astype(np.int32, copy=False) @ weights
    utterances_enc = utterances.astype(np.int32, copy=False) @ weights

    A = np.zeros((n_eval, n_interp, n_order), dtype=np.int8)
    B = np.zeros((n_eval, n_interp, n_order), dtype=np.int8)
    for i in range(n_eval):
        compat = languages_enc == utterances_enc[i]
        A[i] = compat[:, :, true_scenes[i]]
        B[i] = compat[:, :, indices_scenes[i]].sum(-1)
    return A.astype(np.float32), B.astype(np.float32)


def estimate_p_correct(current_state, A, B):
    """
    Estimate E[P(picks true scene)] over the trials precomputed in A, B.
    For each sampled trial, P(correct) = (A·state) / (B·state) — i.e.
    the choose_scene formula evaluated at the true scene. Eval trials
    where the denominator is 0 (current_state predicts the utterance
    for none of the 4 candidates) match choose_scene's uniform-pick
    fallback, contributing 0.25 to the average rather than being
    silently dropped.

    A and B are stored as float32 by precompute_eval_arrays so the
    matmul dispatches to BLAS — this is ~10x faster than the
    broadcast-multiply-then-sum form, which materializes a
    (n_eval, n_interp, n_order) intermediate.
    """
    state_flat = current_state.astype(A.dtype, copy=False).ravel()
    num = A.reshape(A.shape[0], -1) @ state_flat
    den = B.reshape(B.shape[0], -1) @ state_flat
    p_per_trial = np.full(num.shape[0], 0.25, dtype=float)
    mask = den > 0
    p_per_trial[mask] = num[mask] / den[mask]
    return p_per_trial.mean()


def estimate_p_correct_local(current_state, A, B, prior, n_particles):
    """
    Estimate E[P(picks true scene)] for the particles_local model where
    model averaging on each eval trial uses a *fresh* sample of
    n_particles from prior * live_mask(current_state). This matches the
    cognitive interpretation of the model: the learner can only attend
    to n_particles hypotheses on any given hypothetical trial and their
    accuracy reflects what those particles collectively predict. Eval
    trials where no sampled particle predicts the utterance for any of
    the 4 candidates contribute 0.25 (uniform-pick fallback) rather
    than being silently dropped.
    """
    n_eval = A.shape[0]
    flat_state = current_state.flatten()
    live_mask = flat_state > 0
    sampling_weights = prior.flatten() * live_mask
    total = sampling_weights.sum()
    if total == 0:
        # No live hypothesis in the prior's support — the learner falls
        # back to uniform picks on every eval trial.
        return 0.25
    sampling_weights = sampling_weights / total

    particle_indices = np.random.choice(
        flat_state.size, size=(n_eval, int(n_particles)),
        p=sampling_weights, replace=True,
    )
    A_per_trial = np.take_along_axis(
        A.reshape(n_eval, -1), particle_indices, axis=1
    ).mean(axis=1)
    B_per_trial = np.take_along_axis(
        B.reshape(n_eval, -1), particle_indices, axis=1
    ).mean(axis=1)

    p_per_trial = np.full(n_eval, 0.25, dtype=float)
    mask = B_per_trial > 0
    p_per_trial[mask] = A_per_trial[mask] / B_per_trial[mask]
    return p_per_trial.mean()


def run_experiment(trials, interpret_prior, word_order_prior,
                   negative_evidence, noise, return_p_scene=False,
                   noise_type='uniform', feedback_on_false_choices=True,
                   eval_language=None, n_eval_trials=1000,
                   return_p_correct=False, n_particles=None):

    scenes, languages = define_objects()

    if return_p_correct:
        assert eval_language is not None, \
            "eval_language must be provided when return_p_correct=True"
        A_eval, B_eval = precompute_eval_arrays(
            eval_language, scenes, languages, n_eval=n_eval_trials
        )

    # dims: (interpretation function, word order)
    # Contains the probabilities of each language,
    # encoded as a combination of intepretation function
    # and word order
    prior = interpret_prior[:,None] * word_order_prior
    current_state = prior.copy()
    # State to fall back on when current_state has no language with
    # positive weight that can produce the trial's utterance for any of
    # the four candidates. This freezes at whatever current_state held
    # just before learning eliminated the last consistent hypothesis.
    last_compatible_state = current_state.copy()

    def _p_correct(state):
        if noise_type == 'particles_local':
            return estimate_p_correct_local(state, A_eval, B_eval, prior, n_particles)
        return estimate_p_correct(state, A_eval, B_eval)

    # loop through the observations and update the posterior
    # at each timestep
    history_states = [current_state]
    history_choices = []
    p_scenes = []
    # p_correct_general is aligned with history_states: one value per
    # state, starting from the prior and ending at the final posterior.
    p_correct_general = [_p_correct(current_state)] if return_p_correct else []

    for true_scene, indices_scenes, utterance in zip(*trials):

        # get indicator of utterance for each language
        # NOTE: some languages don't use a certain
        # utterance at all!
        # This gives P(scenes | language, utterance)
        # Which is always 0 or 1
        utterance_by_language = (languages == utterance).all(-1)

        # For noise_type='particles_local', the learner's *attention*
        # for this trial is a fresh sample of n_particles hypotheses
        # drawn from prior * live_mask. The same particle set is used
        # for choose_scene's model averaging AND for the kill step, so
        # predictions and updates draw on exactly the hypotheses the
        # agent thought of this trial. State_for_choice is the
        # empirical distribution over those particles.
        sampled_indices = None
        if noise_type == 'particles_local':
            flat_state = current_state.flatten()
            live_mask = flat_state > 0
            sampling_weights = prior.flatten() * live_mask
            sw_total = sampling_weights.sum()
            if sw_total > 0:
                sampling_weights = sampling_weights / sw_total
                sampled_indices = np.random.choice(
                    flat_state.size, size=int(n_particles),
                    p=sampling_weights, replace=True,
                )
                particle_state = np.zeros_like(flat_state)
                np.add.at(particle_state, sampled_indices, 1.0 / len(sampled_indices))
                primary_state = particle_state.reshape(current_state.shape)
            else:
                # current_state has nothing live in the prior's support;
                # fall through to rollback below.
                primary_state = current_state
        else:
            primary_state = current_state

        # If the primary state has a positive-weight language that
        # explains the utterance for one of the candidate scenes, use
        # it for the choice and refresh the rollback target. Otherwise
        # fall back to the last compatible state.
        primary_compat_mass = (
            utterance_by_language[:, :, indices_scenes] * primary_state[:, :, None]
        ).sum()
        if primary_compat_mass > 0:
            state_for_choice = primary_state
            last_compatible_state = primary_state.copy()
        else:
            state_for_choice = last_compatible_state

        chosen_scene, p_scene = choose_scene(
            utterance_by_language,
            indices_scenes,
            state_for_choice,
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
            noise_type,
            feedback_on_false_choices,
            n_particles=n_particles,
            prior=prior,
            sampled_indices=sampled_indices,
        )
        history_states.append(current_state)
        if return_p_correct:
            p_correct_general.append(_p_correct(current_state))

    if return_p_scene and return_p_correct:
        return history_states, history_choices, p_scenes, p_correct_general
    if return_p_correct:
        return history_states, history_choices, p_correct_general
    if return_p_scene:
        return history_states, history_choices, p_scenes
    return history_states, history_choices


def run_experiments(index_true_lang, interpret_prior, word_order_prior, n,
                    negative_evidence, return_p_scene=False, feedback_on_false_choices=True,
                    noise=0, noise_type='uniform', n_particles=None,
                    return_p_correct=False, n_eval_trials=1000):

    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )

    # Use the priors the caller passed in (normalized for safety). The
    # previous implementation silently overwrote them with uniform.
    interpret_prior = np.asarray(interpret_prior, dtype=float)
    interpret_prior = interpret_prior / interpret_prior.sum()
    word_order_prior = np.asarray(word_order_prior, dtype=float)
    word_order_prior = word_order_prior / word_order_prior.sum()

    if displaybar:
        f = IntProgress(min=0, max=n)
        display(f)

    histories = []
    histories_p_correct = []
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
        output = run_experiment(
            trials,
            interpret_prior,
            word_order_prior,
            negative_evidence,
            noise=noise,
            noise_type=noise_type,
            return_p_scene=return_p_scene,
            feedback_on_false_choices=feedback_on_false_choices,
            eval_language=languages[index_lang] if return_p_correct else None,
            n_eval_trials=n_eval_trials,
            return_p_correct=return_p_correct,
            n_particles=n_particles,
        )
        histories.append(output[0])
        if return_p_correct:
            histories_p_correct.append(output[-1])
        if displaybar:
            f.value += 1
    if return_p_correct:
        return histories, histories_p_correct
    return histories


def simulate_full_experiment(n_trials, n_participants, true_languages_setting,
                             simulate_responses, negative_evidence=False,
                             trial_type="unique",
                             noise=0, return_p_scene=False, noise_type='uniform',
                             feedback_on_false_choices=True,
                             return_p_correct=False, n_eval_trials=1000,
                             n_particles=None,
                             interpret_prior=None, word_order_prior=None):
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
    noise: float or None
        The noise of the Bayesian update
        (0 means no noise, 1 means full noise).
        Must be None when noise_type='particles' (the stochasticity is
        set by n_particles instead).
    noise_type: str
        The type of noise to add
        'uniform': likelihood function is smoothed out
        'ignore': a language that is excluded by the data is
            nonetheless kept to 1 with probability "noise"
        'particles': finite-sample posterior — at each trial the
            learner samples n_particles hypotheses from current_state
            and kills the incompatible ones. As current_state
            concentrates, exclusion events naturally become rare.
        'particles_negbin': as 'particles' but the per-trial sample
            size is drawn from a negative binomial (gamma-Poisson)
            with the given (mean, dispersion). variance = mean +
            mean^2/dispersion, so small dispersion gives heavy-tailed
            "bursts" of effort while large dispersion approaches a
            Poisson with mean. Trials with N=0 produce no update.
    n_particles:
        Required when noise_type starts with 'particles'.
        - 'particles': int (per-trial sample size), or 1-D array of
          ints with length n_participants for per-participant values.
        - 'particles_negbin': a (mean, dispersion) pair, or a
          (n_participants, 2) array of pairs for per-participant
          distributions.
    interpret_prior, word_order_prior: array-like or None
        Optional prior weights over interpretation functions
        (length = n_interp) and word orders (length = n_order),
        encoding which hypotheses the learner finds a priori more
        plausible. Any non-negative vector is accepted; it will be
        normalized internally. None (default) means uniform.
    trial_type: str
        The type of trial to simulate
        'unique': each participant sees same trials
        'same': all participants see the same trials
        This only makes sense if true_languages_setting is 'same'
    """

    if return_p_scene:
        assert simulate_responses, "return_p_scene is only available if simulate_responses is True"
    if return_p_correct:
        assert simulate_responses, "return_p_correct is only available if simulate_responses is True"

    scenes, languages, language_interpret, word_orders = define_objects(
        full_output=True
    )

    # Build priors. If the caller passed nothing, default to uniform;
    # otherwise normalize the provided weights so any non-negative
    # vector encoding hypothesis preferences works.
    if interpret_prior is None or word_order_prior is None:
        default_wo, default_int = define_priors(word_orders, language_interpret)
    if interpret_prior is None:
        interpret_prior = default_int
    else:
        interpret_prior = np.asarray(interpret_prior, dtype=float)
        interpret_prior = interpret_prior / interpret_prior.sum()
    if word_order_prior is None:
        word_order_prior = default_wo
    else:
        word_order_prior = np.asarray(word_order_prior, dtype=float)
        word_order_prior = word_order_prior / word_order_prior.sum()

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

    elif true_languages_setting == 'balanced':
        ### Equal number of participants per word order, random interp
        ### within each. Participants are listed in blocks by order:
        ### the first (n_participants/n_orders) get order 0, the next
        ### block gets order 1, and so on.
        n_orders = languages.shape[1]
        assert n_participants % n_orders == 0, (
            f"true_languages_setting='balanced' requires n_participants "
            f"divisible by {n_orders} (number of word orders), got "
            f"n_participants={n_participants}"
        )
        per_order = n_participants // n_orders
        indices_true_language = [
            (np.random.randint(low=0, high=len(languages)), order)
            for order in range(n_orders)
            for _ in range(per_order)
        ]

    else:
        raise ValueError('Setting unknown!')

    if return_p_scene:
        history_p_scenes = []
    if return_p_correct:
        history_p_correct = []

    if trial_type == 'same':
        # all participants see the same trials
        assert true_languages_setting == "same", "trial_type 'same' is only available if true_languages_setting is 'same'"
        trials = generate_trials(
            languages[index_true_language[0]], 
            n=n_trials, 
            scenes=scenes, 
            n_scenes_trial=4
        )

    for i in range(n_participants):

        if noise is None or isinstance(noise, (int, float)):
            noise_i = noise
        else:
            noise_i = noise[i]

        if noise_type == 'particles_negbin':
            arr = np.asarray(n_particles, dtype=float)
            if arr.ndim == 1 and arr.size == 2:
                # single (mean, dispersion) — broadcast to all participants
                n_particles_i = (float(arr[0]), float(arr[1]))
            elif arr.ndim == 2 and arr.shape == (n_participants, 2):
                n_particles_i = (float(arr[i, 0]), float(arr[i, 1]))
            else:
                raise ValueError(
                    f"For noise_type='particles_negbin', n_particles must "
                    f"be a (mean, dispersion) pair or shape "
                    f"({n_participants}, 2), got shape {arr.shape}"
                )
        elif n_particles is None or isinstance(n_particles, (int, float)):
            n_particles_i = n_particles
        else:
            n_particles_i = n_particles[i]

        index_true_language = indices_true_language[i]

        if trial_type == 'unique':
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
                noise_type=noise_type,
                feedback_on_false_choices=feedback_on_false_choices,
                eval_language=languages[index_true_language] if return_p_correct else None,
                n_eval_trials=n_eval_trials,
                return_p_correct=return_p_correct,
                n_particles=n_particles_i,
            )

            # run_experiment returns history_states, history_choices,
            # then optionally p_scenes (if return_p_scene), then optionally
            # p_correct (if return_p_correct) — in that order.
            h_states, h_choices = output[0], output[1]
            extras = list(output[2:])
            if return_p_scene:
                history_p_scenes.append(extras.pop(0))
            if return_p_correct:
                history_p_correct.append(extras.pop(0))

            history_states.append(h_states)
            history_choices.append(h_choices)

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
    if return_p_correct:
        history_p_correct = np.swapaxes(history_p_correct, 0, 1)

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

    if return_p_correct:
        returndict.update({
            'history_p_correct': history_p_correct
        })

    return returndict
