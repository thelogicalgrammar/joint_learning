from model.scripts import weight_model_fit
import argparse
from pprint import pprint
import pickle
from glob import glob

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
    '--extract_summary',
    default=False,
    type=bool,
    help='Whether to extract summary from the pickle files'
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
    
    if args.extract_summary:

        # basepath = "/mnt/c/Users/faust/Documents/joint_learning/param_recovery/"
        basepath = './param_recovery/'

        summaries = list()
        for filepath in glob(basepath+'*.pickle'):

            with open(filepath, 'rb') as openfile:
                ex = pickle.load(openfile)

            # remove probs_orders which is a huuuge value
            del ex['data']['probs_orders'] 

            summary = {
                'true_hyper_ms': ex['data'], 
                'recovered_hyper_ms': ex['samples'].posterior
            }

            summaries.append(summary)

        with open(basepath+'summaries.pickle', 'wb') as openfile:
            pickle.dump(
                summaries,
                openfile
            )
    else:
        weight_model_fit.simulate_parameter_recovery(
            n_trials=args.n_trials,
            n_participants=args.n_participants, 
            recovery_method=args.method,
            n=args.n
        ) 
