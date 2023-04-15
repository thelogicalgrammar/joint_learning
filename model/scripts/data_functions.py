import pandas as pd
import numpy as np
from re import split
from copy import deepcopy
from scipy.stats import rankdata
try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as binomtest

def get_data(datapath=None):

    if datapath is None:
        datapath = '../michael_data/data/results_2021-08-23T12_58_44_175Z_langlearning-v2.csv'

    raw_data = pd.read_csv(datapath)

    # display(raw_data.head())

    # How to remove xiaochen and bob results?
    # raw_data = raw_data.iloc[38:]

    data = raw_data.iloc[:,[0, 2, 6, 9, 11, 12]]

    # this is the same column
    # selection they do in the R script
    data = data.rename(columns={
        k:v
        for k,v in 
        zip(data.columns, 
            ["subject", "counter", "controller", "type", "condition", "response"])
    })

    data['response'] = data['response'].str.replace('%2C', ' ')

    # Exclude the prolific IDs from the data:
    data = data[~(data['condition'] == 'prolific')]
    # Delete `subject` 0 because the `response` columns makes no sense:
    data = data[data['subject'] > 0]

    # just in case we excluded some participant that's not 0 or 1
    # this allows us to use 'rank_partic' as an index
    data['rank_partic'] = rankdata(data['subject'], method='dense') - 1
    
    return data


def replace_values(series, to_replace):
    for k,v in to_replace.items():
        series = series.str.replace(k,str(v))
    return series


def get_analysis_arrays(data, exclude_nonimproving_participants=True):
    """
    Returns
    -------
    Dict
        Dictionary with the following keys:
        - orders
            - shape: (participant, 3)
        - word_order_partic
            - shape: participant
            - The word order for each participant, as a string. E.g., "svo"
        - interpretation_fs_partic
            - shape: (participant, signal, meaning)
            - Binary array
            - Which signal expresses which meaning in each interpretation function
        - scenes_trials
            - shape: (trial, participant, scene, 3)
            - The 4 scenes (3 components) seen by each participant on each trial
        - true_scenes_trials
            - shape:  (trial, participant, 3)
            - True scene (3 components) in each trial for each participant
            - The three scene components are [agent, action, patient]
        - signals
            - shape:  (trial, participant, 3)
            - Utterance (3 signals) seen by each participant in each trial
        - history_choices_indices 
            - shape:  (trial, participant)
            - Index of scene chosen by each participant in each trial 
            - Out of the presented 4 recorded in indices.
    """
    
    word_to_index_dict = {
        word: i
        for i, word in enumerate(['Klin', 'Praz', 'Yabe', 
                                  'Teck', 'Neep', 'Blom', 'Vode'])
    }

    ### Note: for consistency with the model below, 
    ### it's important that the objects receive indices 0 to 3 
    # and the actions indices 4 to 6

    object_to_index_dict = {
        word: i
        for i, word in
        enumerate(['Circle', 'Triangle', 'Heart', 
                   'Square', 'Punch', 'Photo', 'Greet'])
    }

    ### Create array with word order for each participant

    word_order_partic = (
        data[data['condition']=='wordOrderBox']
        ['response']
    ).reset_index(drop=True)

    # subject is 0, verb is 1, object is 2
    orders = replace_values(
        word_order_partic,
        to_replace={
            's': 0,
            'v': 1,
            'o': 2
        }
    )

    orders = pd.DataFrame(
        orders.apply(list).tolist(),
    ).values.astype(int)

    # repeat word order once for each trial
    orders_repeated = np.repeat(
        orders, 
        200,
        0
    )

    ### Create array with true lang for each participant

    data_interpretation_fs = (
        data
        [
            data['condition']
            .isin(['nounMap', 'verbMap'])
        ]
        [['subject', 'condition', 'response', 'rank_partic']]
        .reset_index(drop=True)
    )

    data_interpretation_fs.loc[:,'response'] = (
        data_interpretation_fs
        ['response']
        .apply(
            lambda s: list(
                zip(*[j.split() for j in s.split('maps to')])
            )
        )
    )

    data_interpretation_fs = (
        data_interpretation_fs
        .explode('response')
        .reset_index(drop=True)
    )

    data_interpretation_help = pd.DataFrame(
        data_interpretation_fs['response'].values.tolist(),
        index=data_interpretation_fs.index,
        columns=['meaning', 'word']
    ).reset_index(drop=True)

    # data_interpretation_help.[data_interpretation_help.columns['rank_partic']] = (
    data_interpretation_help.loc['rank_partic'] = (
	    data_interpretation_fs['rank_partic']
	)

    # data_interpretation_help[data_interpretation_help.columns['meaning']] = (
    data_interpretation_help.loc['meaning'] = (
        data_interpretation_help['meaning']
        .replace(object_to_index_dict)
    )
    # data_interpretation_help[data_interpretation_help.columns['word']] = (
    data_interpretation_help.loc['word'] = (
        data_interpretation_help['word']
        .replace(word_to_index_dict)
    )

    n_participants = data_interpretation_help['rank_partic'].max()+1

    # create an array to hold the language
    # used for each participant
    # dims: (participant, word, meaning)
    interpretation_fs_partic = np.zeros((
        n_participants,
        7,
        7
    ))

    # assign 1 to the right combinations
    # of word and meaning
    interpretation_fs_partic[
        data_interpretation_help['rank_partic'],
        data_interpretation_help['word'],
        data_interpretation_help['meaning']
    ] = 1
    
    #### Making `scenes_trials`

    help_df = (
        data
        [data['condition'].str.contains('Sit')]
        [['condition', 'response', 'rank_partic']]
    )

    # this function takes a string like 'HeartPunchSquare' and splits it into three words
    split_into_words = lambda s: split(
        '('+ 
        '|'.join(object_to_index_dict.keys()) 
        +')', 
        s
    )[1::2]

    # help_df contains 4 lines for each participant.
    # The 'response' column is a list of lists, 
    # each sublist corresponding to a trial
    # The 'condition' column specifies which scene is reported
    # of each trial: whether the target or one of the distractors
    help_df.loc[:,'response'] = (
        help_df
        ['response']
        .apply(
            lambda x: [split_into_words(s) for s in x.split()]
        )
    )

    help_df = help_df.explode('response')

    help_df.loc[:,'trial'] = (
        help_df
        .groupby(['rank_partic', 'condition'])
        .cumcount()
    )

    # component_index has the indices of the component for each 
    # scene component in the order: AGENT, VERB, OBJECT
    # Note that the verb always has index 4, 5, or 6
    # and nouns have index 0, 1, 2, or 3
    help_df.loc[:,'component_index'] = (
        help_df
        ['response']
        .apply(
            # this function goes from the list of
            # three scene components in 'response'
            # to their index
            lambda x: [object_to_index_dict[i] for i in x]
        )
    )

    scenes_trials = (
        np.row_stack(help_df['component_index'].values)
        .reshape(n_participants, 4, 200, -1)
    )

    # Change order of axes so that it is (trial, participant, scene, 3)
    scenes_trials = scenes_trials.transpose(2,0,1,3)

    #### Making `true_scenes_trials`
    # Because of the way that the data is stored, the target scene 
    # is always the first one among the four scenes:
    true_scenes_trials = scenes_trials[:,:,0]

    #### Making `signals`

    help_df_target = help_df[help_df['condition'] == 'targetSit']

    # go from scene component indices to the signals used 
    # in the true language of each participant 
    # NOTE: it MUST be relative to the participant
    help_df_target.loc[:,'signals_wt_wordorder'] = (
        help_df_target
        [['rank_partic', 'component_index']]
        .apply(
            lambda x: [
                np.argwhere(
                    interpretation_fs_partic[
                        x['rank_partic'],:,i
                    ]
                )[0,0]
                for i in x['component_index']
            ],
            axis=1
        )
    )

    Xs, _ = np.indices(orders_repeated.shape)

    # NOTE: The reshaping is based on the fact that the rows
    # are organised in the same way as `help_df_target`, 
    # and therefore first by participant index and then by trial.

    signals = (
        np.row_stack(
            help_df_target['signals_wt_wordorder']
        )
        [Xs, orders_repeated]
        .reshape(
            n_participants, 200, -1
        )
    )

    signals = signals.transpose(1,0,2)

    #### Making `history_choices_indices`

    help_df = data[data['condition']=='clickedImage'][['rank_partic', 'response']]
    help_df.loc[:,'response'] = help_df['response'].str.split()
    help_df = help_df.explode('response')
    help_df.loc[:,'trial'] = np.tile(np.arange(200), n_participants)

    # shape: (trial, participant)
    history_choices_indices = replace_values(
        help_df['response'],
        {
            'target': 0,
            'controlSit1': 1,
            'controlSit2': 2,
            'controlSit3': 3
        }
    ).astype(int).values.reshape(n_participants, 200).T
    
    if exclude_nonimproving_participants:
        ######## Exclude participants who didn't improve

        trial_index, participant_index = np.indices(
            history_choices_indices.shape
        )

        df = pd.DataFrame({
            'scaled_trial': trial_index.flatten() / trial_index.max(),
            'partic': participant_index.flatten(),
            'order': np.apply_along_axis(
                lambda x: '|'.join(x.astype(str)), 
                1, 
                np.tile(
                    orders, 
                    (200,1)
                )
            ),
            'right': (history_choices_indices == 0).flatten()
        })

        arr = (
            df
            # get the second half of experiment
            [df['scaled_trial']>0.5]
            # group by participant
            .groupby('partic')
            # get some summary values for
            # each participant
            .agg({
                # within a participant order is fixed,
                # so just get the first one
                'order': 'first',
                # get the total number of correct
                # answers in second half of exp
                'right': 'sum',
                # get number of considered trials
                'scaled_trial': 'count'
            })
        )

        above_chance = (
            arr.apply(
                # run one-sided binomial test
                # for each participant
                lambda x: binomtest(
                    k=int(x['right']), 
                    n=x['scaled_trial'], 
                    p=0.25, 
                    alternative='greater'
                ).pvalue, 
                axis=1
            # check that resulting p-value is
            # significant
            ) < 0.05
        ).values

        orders = orders[above_chance]
        word_order_partic = word_order_partic[above_chance]
        interpretation_fs_partic = interpretation_fs_partic[above_chance]
        scenes_trials = scenes_trials[:,above_chance]
        true_scenes_trials = true_scenes_trials[:,above_chance]
        signals = signals[:,above_chance]
        history_choices_indices = history_choices_indices[:,above_chance]
    
    return {
        'orders': orders,
        'word_order_partic': word_order_partic,
        'interpretation_fs_partic': interpretation_fs_partic,
        'scenes_trials': scenes_trials,
        'true_scenes_trials': true_scenes_trials,
        'signals': signals,
        'history_choices_indices': history_choices_indices
    }


def get_first_n_trials(analysis_arrays, n_trials='all'):
    
    if n_trials=='all':
        # get all the values in the 
        # indexing below
        n_trials = None
    
    return_arrays = deepcopy(analysis_arrays)
    
    # just change the arrays with a trial dimension
    return_arrays['scenes_trials'] = return_arrays[
        'scenes_trials'
    ][:n_trials]
    return_arrays['true_scenes_trials'] = return_arrays[
        'true_scenes_trials'
    ][:n_trials]
    return_arrays['signals'] = return_arrays[
        'signals'
    ][:n_trials]
    return_arrays['history_choices_indices'] = return_arrays[
        'history_choices_indices'
    ][:n_trials]
    
    return return_arrays
