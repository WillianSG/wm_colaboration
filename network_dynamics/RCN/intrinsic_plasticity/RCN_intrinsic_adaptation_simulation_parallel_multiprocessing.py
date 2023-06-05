# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bernoulli Institute
"""
import argparse
import atexit
import copy
import glob
import itertools
import random
import shutil
import signal
import sys
import time
import warnings
from functools import partial
import os

import pandas as pd
import pathos
import tqdm_pathos
from brian2 import Hz
from dill import PicklingWarning, UnpicklingWarning
from matplotlib.ticker import FormatStrFormatter
from numpy import VisibleDeprecationWarning
from tqdm import TqdmWarning
from tqdm.auto import tqdm

from helper_functions.other import *
from helper_functions.population_spikes import count_ps
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from plotting_functions.plot_thresholds import *
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin

warnings.simplefilter("ignore", PicklingWarning)
warnings.simplefilter("ignore", UnpicklingWarning)
warnings.simplefilter("ignore", VisibleDeprecationWarning)


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def product_dict_to_list(**kwargs):
    from itertools import product

    vals = kwargs.values()

    out_list = []
    for instance in product(*vals):
        out_list.append(list(instance))

    return out_list


def run_sim(params, plot=True, progressbar=True, seed_init=None, low_memory=True):
    warnings.filterwarnings("ignore", category=TqdmWarning)

    if (plot == True and low_memory == True):
        raise ValueError('plotting is only possible with low_memory=True')

    pid = os.getpid()
    # print(f'RUNNING worker {pid} with params: {params}')

    if isinstance(params, list):
        ba = params[0]
        i_e_w = params[1]
        e_i_w = params[2]
        e_e_maxw = params[3]
        i_freq = params[4]
        cue_percentage = params[5]
        cue_time = params[6]
        num_attractors = int(params[7])
        num_cues = int(params[8])
        attractor_size = int(params[9])
        network_size = int(params[10])
    elif isinstance(params, dict) or isinstance(params, pd.Series):
        ba = params['background_activity']
        i_e_w = params['i_e_weight']
        e_i_w = params['e_i_weight']
        e_e_maxw = params['e_e_max_weight']
        i_freq = params['i_frequency']
        cue_percentage = params['cue_percentage']
        cue_time = params['cue_length']
        num_attractors = int(params['num_attractors'])
        num_cues = int(params['num_cues'])
        attractor_size = int(params['attractor_size'])
        network_size = int(params['network_size'])

    worker_id = params[11] if len(params) == 12 else None

    import sympy
    a, s, n = sympy.symbols('a s n')
    expr = a * (s + 16) <= n
    if not expr.subs([(a, num_attractors), (s, attractor_size), (n, network_size)]):
        print(
            f'{num_attractors} attractors of size {attractor_size} cannot fit into a network of {network_size} neurons.')
        choice = input('1- Smaller attractors\n2- Fewer attractors\n3- Larger network\n')
        if choice == '1':
            asol = sympy.solveset(expr.subs([(a, num_attractors), (n, network_size)]), s, sympy.Integers)
            attractor_size = asol.sup
            # attractor_size = floor((network_size - num_attractors * 16) / num_attractors)
            print(f'Optimal attractor size: {attractor_size}')
        if choice == '2':
            asol = sympy.solveset(expr.subs([(s, attractor_size), (n, network_size)]), a, sympy.Integers)
            num_attractors = asol.sup
            print(f'Optimal number of attractors: {num_attractors}')
        if choice == '3':
            asol = sympy.solveset(expr.subs([(a, num_attractors), (s, attractor_size)]), n, sympy.Integers)
            network_size = asol.inf
            print(f'Optimal network size: {network_size}')

    rcn = RecurrentCompetitiveNet(
        stim_size=attractor_size,
        plasticity_rule='LR4',
        parameter_set='2.2',
        seed_init=seed_init,
        low_memory=low_memory)

    plastic_syn = False
    plastic_ux = True
    rcn.E_E_syn_matrix_snapshot = False
    rcn.w_max = e_e_maxw * mV  # for param. 2.1: 10*mV
    rcn.w_e_i = e_i_w * mV  # 3 mV default
    rcn.w_i_e = i_e_w * mV  # 10 mV default
    rcn.spont_rate = ba * Hz

    # -- intrinsic plasticity setup (Vth_e_decr for naive / tau_Vth_e for calcium-based)
    rcn.tau_Vth_e = 0.1 * second  # 0.1 s default
    # rcn.Vth_e_init = -51.6 * mV           # -52 mV default
    rcn.k = 4  # 2 default

    rcn.net_init()

    # -- synaptic augmentation setup
    rcn.U = 0.2  # 0.2 default

    rcn.set_stimulus_i(
        stimulus='flat_to_I',
        frequency=i_freq * Hz)  # default: 15 Hz

    # --folder for simulation results
    rcn.net_sim_data_path = make_timestamped_folder(os.path.join(tmp_folder, f'{pid}'))

    # output .txt with simulation summary.
    _f = os.path.join(rcn.net_sim_data_path, 'simulation_summary.txt')

    attractors_list = []
    for i in range(num_attractors):
        if i * (rcn.stim_size_e + 16) + rcn.stim_size_e > len(rcn.E):
            print(
                f'{num_attractors} attractors of size {rcn.stim_size_e} cannot fit into a network of {len(rcn.E)} neurons.  Instantiating {i} attractors instead.')
            num_attractors = i
            break
        stim_id = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=i * (64 + 16))
        rcn.set_potentiated_synapses(stim_id)
        attractors_list.append([f'A{i}', stim_id])

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    # generate shuffled attractors
    '''Attractor cueing order:
        - Attractor name
        - Neuron IDs belonging to attractor
        - Length of triggered window
        - Generic stimulus
            - % of neurons targeted within attractor
            - IDs of targeted neurons
            - (start time of GS, end time of GS)'''
    attractors_cueing_order = [copy.deepcopy(random.choice(attractors_list))]
    for i in range(num_cues - 1):
        atr_sel = copy.deepcopy(random.choice(attractors_list))
        while attractors_cueing_order[i][0] == atr_sel[0]:
            atr_sel = copy.deepcopy(random.choice(attractors_list))
        attractors_cueing_order.append(atr_sel)

    # generate random reactivation times in range
    for a in attractors_cueing_order:
        a.append(random.uniform(4, 4))

    overall_sim_time = len(attractors_cueing_order) * cue_time + sum([a[2] for a in attractors_cueing_order])
    pbar = tqdm(total=overall_sim_time, disable=not progressbar, unit='sim s', leave=True, position=worker_id,
                desc=f'WORKER {worker_id}',
                bar_format="{l_bar}{bar}|{n:.2f}/{total:.2f} [{elapsed}<{remaining},{rate_fmt}{postfix}]")
    pbar.set_description(f'WORKER {worker_id} with params {params}')
    time.sleep(0.1)
    pbar.update(0)
    for a in attractors_cueing_order:
        gs = (cue_percentage, a[1], (rcn.net.t / second, rcn.net.t / second + cue_time))
        act_ids = rcn.generic_stimulus(
            frequency=rcn.stim_freq_e,
            stim_perc=gs[0],
            subset=gs[1])
        rcn.run_net(duration=cue_time)
        rcn.generic_stimulus_off(act_ids)
        # # wait for 2 seconds before cueing next attractor
        rcn.run_net(duration=a[2])
        a.append(gs)
        pbar.update(cue_time + a[2])

        # after each cue block dump the monitored values to a file and clear the memory
        if low_memory:
            rcn.dump_monitors_to_file()

    rcn.load_monitors_from_file()

    # -- calculate score
    atr_ps_counts = count_ps(rcn=rcn, attractor_cueing_order=attractors_cueing_order)

    trig, spont, accuracy, stability = compute_ps_score(atr_ps_counts, attractors_cueing_order)

    if plot:
        title_addition = f'BA {ba} Hz, GS {cue_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
        filename_addition = f'_BA_{ba}_GS_{cue_percentage}_W_{i_e_w}_Hz_{i_freq}'

        plot_x_u_spks_from_basin(
            path=rcn.net_sim_data_path,
            filename='x_u_spks_from_basin' + filename_addition,
            title_addition=title_addition,
            rcn=rcn,
            attractor_cues=attractors_cueing_order,
            pss_categorised=atr_ps_counts,
            num_neurons=len(rcn.E),
            show=plot)

        # plot_thresholds(
        #     path=rcn.net_sim_data_path,
        #     file_name='thresholds' + filename_addition,
        #     rcn=rcn,
        #     attractors=attractors_list,
        #     show=True)

    # print(f'FINISHED worker {pid} triggered: {trig}, spontaneous: {spont}, score: {score}')
    # print(atr_ps_counts)

    # cleanup
    shutil.rmtree(rcn.net_sim_data_path)
    del rcn
    pbar.close()

    return params, (stability, accuracy, trig, spont)


if __name__ == '__main__':
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(tmp_folder)

    parser = argparse.ArgumentParser(
        description='Parallel sweep for parameters of RCN model with STSP and intrinsic plasticity.  '
                    'When --parameter_distribution is "grid" the first parameter is the start value and the second the end value.'
                    'When --parameter_distribution is "gaussian" the first parameter is the mean value and the second the standard deviation.')
    parser.add_argument('-ba', '--background_activity', type=float, default=[15], nargs='+',
                        help='Level of background activity in Hz')
    parser.add_argument('-i_e_w', '--i_e_weight', type=float, default=[10], nargs='+',
                        help='Weight of I-to-E synapses in mV')
    parser.add_argument('-e_i_w', '--e_i_weight', type=float, default=[3], nargs='+',
                        help='Weight of E-to-I synapses in mV')
    parser.add_argument('-e_e_maxw', '--e_e_max_weight', type=float, default=[10], nargs='+',
                        help='Maximum weight of E-to-E synapses in mV')
    parser.add_argument('-i_freq', '--i_frequency', type=float, default=[20], nargs='+',
                        help='Frequency of I input in Hz')
    parser.add_argument('-cue_perc', '--cue_percentage', type=float, default=[100], nargs='+',
                        help='Percentage of neurons in the attractor to be stimulated')
    parser.add_argument('-cue_len', '--cue_length', type=float, default=[1], nargs='+',
                        help='Duration of the cue in seconds')
    parser.add_argument('-num_attractors', '--num_attractors', type=int, default=[3], nargs='+',
                        help='Number of attractors to be generated')
    parser.add_argument('-atr_size', '--attractor_size', type=int, default=[64], nargs='+',
                        help='Number of neurons in each attractor')
    parser.add_argument('-net_size', '--network_size', type=int, default=[256], nargs='+',
                        help='Number of neurons in the network')
    parser.add_argument('-num_cues', '--num_cues', type=int, default=[10], nargs='+',
                        help='Number of cues given to the network')
    parser.add_argument('-num_par', '--num_parameters', type=int, default=2,
                        help='Number of parameters to pick from each distribution')
    parser.add_argument('-param_dist', '--parameter_distribution', type=str, choices=['grid', 'gaussian'],
                        default='grid', help='Distribution of parameters')
    parser.add_argument('-cv', '--cross_validation', type=int, default=2, help='Number of cross validation folds')
    parser.add_argument('-plot', '--plot', type=bool, default=False, help='Plot the results')
    args = parser.parse_args()

    if args.parameter_distribution == 'grid':
        param_dist = np.linspace
        print(
            f'Sampling {args.num_parameters} parameters from grid distribution and validating them {args.cross_validation} times')
    elif args.parameter_distribution == 'gaussian':
        param_dist = np.random.normal
        print(
            f'Sampling {args.num_parameters} parameters from gaussian distribution and validating them {args.cross_validation} times')

    num_cpus = pathos.multiprocessing.cpu_count()
    cv = args.cross_validation
    param_grid = {
        'background_activity': param_dist(args.background_activity[0], args.background_activity[1],
                                          args.num_parameters) if len(
            args.background_activity) > 1 else args.background_activity,
        'i_e_weight': param_dist(args.i_e_weight[0], args.i_e_weight[1], args.num_parameters) if len(
            args.i_e_weight) > 1 else args.i_e_weight,
        'e_i_weight': param_dist(args.e_i_weight[0], args.e_i_weight[1], args.num_parameters) if len(
            args.e_i_weight) > 1 else args.e_i_weight,
        'e_e_max_weight': param_dist(args.e_e_max_weight[0], args.e_e_max_weight[1], args.num_parameters) if len(
            args.e_e_max_weight) > 1 else args.e_e_max_weight,
        'i_frequency': param_dist(args.i_frequency[0], args.i_frequency[1], args.num_parameters) if len(
            args.i_frequency) > 1 else args.i_frequency,
        'cue_percentage': param_dist(args.cue_percentage[0], args.cue_percentage[1], args.num_parameters) if len(
            args.cue_percentage) > 1 else args.cue_percentage,
        'cue_length': param_dist(args.cue_length[0], args.cue_length[1], args.num_parameters) if len(
            args.cue_length) > 1 else args.cue_length,
        'num_attractors': param_dist(args.num_attractors[0], args.num_attractors[1], args.num_parameters) if len(
            args.num_attractors) > 1 else args.num_attractors,
        'num_cues': param_dist(args.num_cues[0], args.num_cues[1], args.num_parameters) if len(
            args.num_cues) > 1 else args.num_cues,
        'attractor_size': param_dist(args.attractor_size[0], args.attractor_size[1], args.num_parameters) if len(
            args.attractor_size) > 1 else args.attractor_size,
        'network_size': param_dist(args.network_size[0], args.network_size[1], args.num_parameters) if len(
            args.network_size) > 1 else args.network_size,
    }
    for v in param_grid.values():
        v.sort()
    sweeped_params = [(k, i) for i, (k, v) in enumerate(param_grid.items()) if len(v) > 1]
    print(param_grid)

    # # ------ TODO debug
    # default_params.insert(4, 0.1)
    # default_params[5] = 4
    # default_params[6] = 5
    # p, s = run_sim(default_params, plot=False, low_memory=True)
    # 0 / 0
    # # ----------------

    param_grid_pool = list(
        itertools.chain.from_iterable(map(copy.copy, product_dict_to_list(**param_grid)) for _ in range(cv)))
    estimate_search_time(partial(run_sim, low_memory=False, plot=True), param_grid_pool, cv, num_cpus=num_cpus)
    # hack to keep track of process ids
    for i, g in enumerate(param_grid_pool):
        g.append(i % num_cpus)

    chunked_param_grid = list(chunks(param_grid_pool, num_cpus))

    # with pathos.multiprocessing.ProcessPool(num_cpus) as p:
    #     results = p.map(partial(run_sim, plot=False), param_grid_pool)
    results = tqdm_pathos.map(partial(run_sim, plot=False, progressbar=os.isatty(sys.stdout.fileno())), param_grid_pool,
                              n_cpus=num_cpus)

    # if len(sweeped_params) > 1:
    #     res_unsweeped_removed = []
    #     for r in results:
    #         res_tmp = []
    #         for p in sweeped_params:
    #             res_tmp.append(r[0][p[1]])
    #         res_unsweeped_removed.append((res_tmp, r[1]))
    # else:
    #     res_unsweeped_removed = results
    #
    # res_unsweeped_removed = results

    sweeped_param_names = [p[0] for p in sweeped_params]

    # create a dataframe with the raw results
    df_results = pd.DataFrame(columns=list(param_grid.keys()) + ['stability', 'accuracy', 'triggered', 'spontaneous'])
    for r in results:
        df_results.loc[len(df_results)] = r[0][:-1] + [*r[1]]

    # aggregate results by the sweeped parameters
    if not sweeped_param_names:
        sweeped_param_names = list(df_results.columns[:-4])
    df_results_aggregated = df_results.copy()
    df_results_aggregated = df_results_aggregated.groupby(sweeped_param_names).mean().reset_index()
    df_results_aggregated.sort_values(by=sweeped_param_names, ascending=True, inplace=True)
    # df_results_aggregated['weighted_score'] = df_results_aggregated['score'] * df_results_aggregated['triggered']
    df_results_aggregated.to_csv(f'{tmp_folder}/results.csv')

    # print best results in order
    print(
        df_results_aggregated[sweeped_param_names + ['stability', 'accuracy', 'triggered', 'spontaneous']].sort_values(
            by='stability',
            ascending=False))
    print(
        df_results_aggregated[sweeped_param_names + ['stability', 'accuracy', 'triggered', 'spontaneous']].sort_values(
            by='accuracy',
            ascending=False))

    df_results_aggregated['sweeped'] = df_results_aggregated[sweeped_param_names].apply(
        lambda x: ', '.join(x.apply('{:,.2f}'.format)), axis=1)
    # df_results_aggregated['sweeped'] = df_results_aggregated['sweeped'].astype(str)

    # plot results for cue time
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    df_results_aggregated.plot(kind='bar', ax=axes[0], x='sweeped', y='accuracy', legend=False)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel(sweeped_param_names)
    axes[0].set_xticklabels(df_results_aggregated['sweeped'], rotation=45)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[0].set_title('Accuracy for sweeped parameters')

    df_results_aggregated.plot(kind='bar', ax=axes[1], x='sweeped', y='stability', legend=False)
    axes[1].set_ylabel('Stability')
    axes[1].set_xlabel(sweeped_param_names)
    axes[1].set_xticklabels(df_results_aggregated['sweeped'], rotation=45)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].set_title('Stability for sweeped parameters')

    df_results_aggregated.plot(kind='bar', ax=axes[2], x='sweeped', y=['triggered', 'spontaneous'], color=['g', 'r'])
    axes[2].set_ylabel('Reactivations')
    axes[2].set_xlabel(sweeped_param_names)
    axes[2].set_xticklabels(df_results_aggregated['sweeped'], rotation=45)
    axes[2].set_title('Reactivations for sweeped parameters')

    fig.suptitle(f'Results', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'{tmp_folder}/score.png')
    fig.show()

    # TODo find best combined accuracy+stability
    # Run model with the best parameters and plot output
    best_params = df_results_aggregated.loc[df_results_aggregated.stability.idxmax()]
    best_score = best_params['stability']
    best_params = best_params.drop(['stability', 'accuracy', 'triggered', 'spontaneous', 'sweeped']).to_dict()
    print(f'Best parameters: {best_params}')
    run_sim(best_params, plot=True, low_memory=False)

    while True:
        save = input('Save results? (y/n)')
        if save == 'y':
            save_folder = f'SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
            os.makedirs(save_folder)
            os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
            os.rename(f'{tmp_folder}/results.csv', f'{save_folder}/results.csv')

            save2 = input('Saved overall results.  Also save individual plots?: (y/n)')
            if save2 == 'y':
                for f in glob.glob(f'{tmp_folder}/**/*.png', recursive=True):
                    pid = os.path.normpath(f).split(os.path.sep)[1]
                    os.makedirs(f'{save_folder}/{pid}', exist_ok=True)
                    os.rename(f, f'{save_folder}/{pid}/{os.path.split(f)[1]}')

            cleanup()
            break
        elif save == 'n':
            cleanup()
            break
        else:
            print("Please enter 'y' or 'n'")
