from model.scripts import weight_model_fit
import argparse
from pprint import pprint

parser = argparse.ArgumentParser(
    prog = 'Run parameter recovery simulations',
)

parser.add_argument(
    '--method',
    choices=['hmc', 'variational'],
    default='variational',
    type=str,
    help='Method of model fitting.'
)

parser.add_argument(
    '--n',
    default=0,
    type=int,
    help='To append at the end of the file'
)

parser.add_argument(
    '--n_trials',
    default=150,
    type=int,
    help='Number of trials to simulate for each participant'
)

parser.add_argument(
    '--n_participants',
    default=150,
    type=int,
    help='Number of participants to simulate for each experiment'
)

if __name__=='__main__':

    args = parser.parse_args()

    print('Arguments passed to the job: ')
    pprint(args)
    
    weight_model_fit.simulate_parameter_recovery(
        n_trials=args.n_trials,
        n_participants=args.n_participants, 
        recovery_method=args.method,
        n=args.n
    ) 
