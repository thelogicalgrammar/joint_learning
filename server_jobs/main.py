from model.scripts import weight_model_fit
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(
    prog = 'Weight model fit',
    description = 'Fits the weight model with the SLURM scheduler'
)

parser.add_argument(
    '--participant_exclusion',
    default=True,
    type=bool,
    help='Whether to run participant exclusion for non-learning participants'
)

parser.add_argument(
    '--method',
    choices=['hmc', 'variational', 'jax'],
    default='hmc',
    type=str,
    help='Method of model fitting.'
)

parser.add_argument(
    '--first_n_trials',
    default='all',
    help='How many trials to fit (from 0 to specified [excluded])'
)

parser.add_argument(
    '--softmax_choice',
    default=True,
    type=bool,
    help='Whether to add softmax estimation for decision'
)

parser.add_argument(
    '--hierarchicallearningweights',
    default=False,
    type=bool,
    help=(
        'Whether to add by-participant hierarchical '
        'structure on learning weights. '
        'NOT WORKING'
    )
)

parser.add_argument(
    '--draws',
    default=1,
    type=int,
    help='How many actual samples to take'
)

parser.add_argument(
    '--tune',
    default=1,
    type=int,
    help='How many tuning samples to take'
)

parser.add_argument(
    '--cores',
    default=4,
    type=int,
    help='How many cores (chains) to use in parallel'
)

parser.add_argument(
    '--datapath',
    default='../data.csv',
    type=str,
    help='Where to look for data (relative to main project directory'
)

parser.add_argument(
    '--outputfile_append',
    default='',
    type=str,
    help='String to append to the end of the stored trace'
)

if __name__=='__main__':

    args = parser.parse_args()

    print('Arguments passed to the job: ')
    pprint(args)
    
    weight_model_fit.get_and_fit_data(
        participant_exclusion=args.participant_exclusion, 
        method=args.method,
        first_n_trials=args.first_n_trials,
        fit_kwargs={
            'draws': args.draws,
            'tune': args.tune,
            'cores': args.cores
        },
        model_kwargs={
            'hierarchicallearningweights': args.hierarchicallearningweights,
            'softmax_choice': args.softmax_choice
        },
        datapath=args.datapath,
        outputfile_append=args.outputfile_append,
    )
