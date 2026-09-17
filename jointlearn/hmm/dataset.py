"""
Load experimental data: signals (T, P, 3), candidate scenes (T, P, 4, 3),
choices (T, P), true word-order per ppt (= the between-subjects condition),
accuracy, and a few derived display constants (per-meaning colours,
word/meaning short names, the word-to-meaning ground-truth `inv_true`).

Imported as `dataset` (usually aliased RP) by the pipeline scripts. The module
runs the upstream CSV through `jointlearn.data.get_data` at import
time, so the first import is the slow one (~5-10s) and downstream imports
are cached.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

import numpy as np

from jointlearn import simulation as sf
from jointlearn.data import get_data, get_analysis_arrays

DATA_PATH = PROJECT_ROOT / 'data' / 'langlearning_v2_anonymized.csv'   # see data/README.md

# ---- display constants ----
order_labels = ['SVO', 'SOV', 'VSO', 'VOS', 'OSV', 'OVS']
order_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b']                          # 6 orders
order_labels_full = order_labels + ['UNK']                     # incl UNK_O
order_colors_full = order_colors + ['#dcdcdc']
# typological groups by initial constituent (indices into order_labels)
TYPOLOGY = {'S-initial': [0, 1], 'V-initial': [2, 3], 'O-initial': [4, 5]}
meaning_abbr = ['Cir', 'Tri', 'Hea', 'Squ', 'Pun', 'Pho', 'Gre']
word_names = ['Klin', 'Praz', 'Yabe', 'Teck', 'Neep', 'Blom', 'Vode']
MEANING_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2']            # 7 meanings

# ---- data ----
data = get_data(datapath=str(DATA_PATH))
arrays = get_analysis_arrays(data, exclude_nonimproving_participants=True)
scenes, languages, language_interpret, word_orders = sf.define_objects(full_output=True)
signals = np.asarray(arrays['signals'])                 # (T, P, 3)
cand_mean = np.asarray(arrays['scenes_trials'])         # (T, P, 4, 3)
choices = np.asarray(arrays['history_choices_indices']) # (T, P); 0 = correct
T, P = choices.shape
accuracy = (choices == 0).mean(axis=0)
# mean2sig[p] is a permutation meaning -> signal; inv_true is its inverse.
mean2sig = np.argmax(arrays['interpretation_fs_partic'], axis=1)
inv_true = np.argsort(mean2sig, axis=1)

_ROLE = {'s': 0, 'v': 1, 'o': 2}
_ORDER_IDX = {tuple(wo): i for i, wo in enumerate(word_orders)}
true_order = np.array([_ORDER_IDX[tuple(_ROLE[c] for c in s)]
                       for s in arrays['word_order_partic']])

# ---- simulated dataset override (parameter recovery) ----
# EA_SIM_DATA=<file.npz> replaces the observed choices by simulated ones
# (written by simulate.save_sim_dataset / 08_param_recovery.py) on the SAME
# stimuli, participants and conditions, so 01_fit.py and every
# downstream script run unchanged on the synthetic data.
SIM_DATA = os.environ.get('EA_SIM_DATA')
if SIM_DATA:
    _sim = np.load(SIM_DATA)
    if _sim['choices'].shape != choices.shape:
        raise SystemExit(f'EA_SIM_DATA={SIM_DATA}: choices {_sim["choices"].shape} '
                         f'do not match the loaded data {choices.shape}')
    choices = _sim['choices'].astype(choices.dtype)
    accuracy = (choices == 0).mean(axis=0)
    print(f'[data] using simulated choices from {SIM_DATA}', flush=True)
