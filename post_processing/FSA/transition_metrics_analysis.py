# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../')))

if sys.platform in ['linux', 'win32']:
    sys.path.append(os.path.join(root, 'helper_functions', 'sFSA'))
    from sFSA import sFSA
else:
    from helper_functions.sFSA import sFSA

'''
    Arguments from parser here don't matter since object instance is only
used for its methods to parse data.
'''
parser = argparse.ArgumentParser(description = 'sFSA')

# General parameters.
parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
parser.add_argument('--e_e_max_weight', type = float,   default = 9.00,   help = 'Within attractor weight (mV).')
parser.add_argument('--W_ei', type = float,             default = 1.50,   help = 'E-to-I weight (mV).')
parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')

# State transition specific parameters.
parser.add_argument('--w_acpt', type = float,           default = 2.0,   help = 'Weight in synapses to GO state (mV).')
parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
parser.add_argument('--thr_GO_state', type = float,     default = -47.00,  help = 'Threshold for Vth gated synapses (mV).')
parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
parser.add_argument('--delay_A2B', type = float,     default = 0.525,  help = 'Connection delay of blue synapses (s).')

# - Initializing sFSA.

args = parser.parse_args()

fsa = {
    'S': ['A'],
    'I': ['0'],
    'T': []
}

sFSA_model = sFSA(fsa, args)

# - Reading data from listed folders.

target_dirs = ['5', '6', '9', 'I', 'III']

'''
Assuming here that 'start_twindow', 'input_twindow', 'free_activity' are the same for all sims.
'''
params_filter = ['ba_rate', 'e_e_max_weight', 'W_ei', 'W_ie', 'inh_rate', 'gs_percent', 'w_acpt', 'w_trans', 'thr_GO_state', 'p_A2GO', 'delay_A2GO', 'p_A2B', 'delay_A2B']

# - Initiazling params./metrics map.

parameters = {}

for param in params_filter:

    parameters[param] = -1

dir_params_map = {}

for t_dir in target_dirs:

    dir_params_map[t_dir] = parameters.copy() # init.

    # Reading parameter change simulation set dir.

    sims_dir = os.listdir(os.path.join(root, 'results', t_dir))

    correct = 0
    null = 0
    multiple_active = 0
    no_trans = 0

    params_read = False

    for sim_dir in sims_dir:                            # Reading sample simulation dir.

        # Getting parameters values (same for all).

        sim_dir_path = os.path.join(os.path.join(root, 'results', t_dir, sim_dir))

        if not params_read:                             # same for all subdir.

            parameters_txt = open(os.path.join(sim_dir_path, 'parameters.txt'))

            lines = parameters_txt.readlines()

            for line in lines:

                if line.split(': ')[0] in params_filter:

                    param = line.split(': ')[0].replace(' s', '').replace('\n', '')
                    val = line.split(': ')[1].replace(' s', '').replace('\n', '')

                    dir_params_map[t_dir][param] = np.round(float(val), 3)

            params_read = True

        # Computing metrics.

        state_history = sFSA_model.computeStateHistory(sim_dir_path)

        if state_history[0] == 'two':
            state_history[0] = 'A'

        if state_history == ['A', 'two']:
            multiple_active += 1
        elif state_history == ['A', 'null']:
            null += 1
        elif state_history == ['A', 'B']:
            correct += 1
        elif state_history == ['A', 'A']:
            no_trans += 1
        else:
            pass

    # Computing transition outcome metrics.

    correct = np.round(correct/len(sims_dir)*100, 2)
    multiple_active = np.round(multiple_active/len(sims_dir)*100, 2)
    null = np.round(null/len(sims_dir)*100, 2)
    no_trans = np.round(no_trans/len(sims_dir)*100, 2)

    # Updating folder - set of params map.

    dir_params_map[t_dir]['correct'] = correct
    dir_params_map[t_dir]['null'] = null
    dir_params_map[t_dir]['two'] = multiple_active
    dir_params_map[t_dir]['no_trans'] = no_trans

    print(f'{t_dir} - correct: {correct}%, multiple active: {multiple_active}%, null: {null}%, no trans.: {no_trans}% ({correct+multiple_active+null+no_trans})\n')

# - Removing constant params across all simulations.

params_dir = dir_params_map.keys()

for param in params_filter:

    _ = []

    for dir in params_dir:

        _.append(dir_params_map[dir][param])

    if len(list(set(_))) == 1:                          # param is not unique.

        for dir in params_dir:

            del dir_params_map[dir][param]

# - Creating dataframe and finding params set with single parameter change.

df = pd.DataFrame(dir_params_map)

print(df)

df.to_csv('sFSA_simulation_sets_metrics.csv')

columns = list(df.columns)
rows = list(df.index)

cols_2_compare = []

for col1 in columns:

    _ = [col1]

    _varying_param = ''

    for col2 in columns:

        if col1 != col2:

            col_dif = df[col1].iloc[:-4] - df[col2].iloc[:-4]
            col_dif = np.count_nonzero(list(col_dif))

            if col_dif == 1:

                if _varying_param == '':
                    _.append(col2)
                    col_dif = df[col1].iloc[:-4] - df[col2].iloc[:-4]
                    _varying_param = col_dif[col_dif != 0.0].bfill(axis=0).index[0]
                else:
                    col_dif = df[col1].iloc[:-4] - df[col2].iloc[:-4]
                    _varying_param_ = col_dif[col_dif != 0.0].bfill(axis=0).index[0]

                    if _varying_param_ == _varying_param:
                        _.append(col2)
    
    aux = _.copy()
    aux.reverse()
    append = True

    for i in cols_2_compare:

        intersec = len(list(set(_) & set(i)))

        if intersec == len(_):
            append = False

    if aux not in cols_2_compare and len(_) > 1 and append:

        cols_2_compare.append(_)

print(cols_2_compare)

plot_name_id = 0

for cols in cols_2_compare:

    aux_pd = {}
    col_param = {}

    col_dif = df[cols[0]].iloc[:-4] - df[cols[1]].iloc[:-4]
    varying_param = col_dif[col_dif != 0.0].bfill(axis=0).index[0]

    sorted_df = df.sort_values(by = varying_param, axis = 1)

    sorted_cols = list(sorted_df.columns)

    _scs = []

    for _sc_i in sorted_cols:

        if _sc_i in cols:

            _scs.append(_sc_i)

    for col in _scs:

        aux_pd[col] = sorted_df[col].iloc[len(rows)-4:]

        col_param[col] = sorted_df[col][varying_param]

    aux_pd = pd.DataFrame(aux_pd)

    _metrics = list(aux_pd.index)
    _sim_sets = list(aux_pd.columns)

    _ = {}
    index = []

    for metric in _metrics:

        _[metric] = []

        for sim_set in _sim_sets:

            _[metric].append(aux_pd[sim_set][metric])

            _indx = f'{sim_set}\n  {varying_param}: {col_param[sim_set]}'

            if _indx not in index:
                index.append(_indx)

    _df = pd.DataFrame(_, index = index)
    ax = _df.plot.bar(rot = 0)

    ax.set_xlabel('parameter set folder')
    ax.set_ylabel('simulations (%)')
    ax.set_yticks(np.arange(0, 110, 10))

    plt.xticks(fontsize=7.5)
    
    plt.legend(framealpha = 0.0)

    plt.tight_layout()

    plt.savefig(f'transition_metrics-{plot_name_id}.png')

    plot_name_id += 1