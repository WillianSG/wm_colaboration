# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: CogniGron
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


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass


def get_default(param):
    for a in parser._actions:
        if a.dest == param:
            return a.default[0]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def product_dict_to_list(dic):
    from itertools import product

    out_list = []
    for instance in product(*dic.values()):
        out_list.append(dict(zip(dic.keys(), instance)))

    return out_list


def is_pycharm():
    return os.getenv("PYCHARM_HOSTED") != None


if is_pycharm():
    from helper_functions.other import *
    from plotting_functions.plot_thresholds import *
    from helper_functions.recurrent_competitive_network import run_rcn
    from helper_functions.telegram_notify import TelegramNotify
else:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__, "../")))
    sys.path.append(os.path.join(root, "helper_functions"))
    from other import *
    from plot_thresholds import *
    from recurrent_competitive_network import run_rcn
    from telegram_notify import TelegramNotify

warnings.simplefilter("ignore", PicklingWarning)
warnings.simplefilter("ignore", UnpicklingWarning)
warnings.simplefilter("ignore", VisibleDeprecationWarning)

if __name__ == "__main__":
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    parser = argparse.ArgumentParser(
        description="Parallel sweep for parameters of RCN model with STSP and intrinsic plasticity.\n"
                    "Passing one value for parameter will default to using Gaussian distribution with that value as mu"
                    "Passing two values for parameter will default to using grid distribution with those values lower and upper bounds"
    )
    parser.add_argument(
        "-background_activity",
        type=float,
        default=[15],
        nargs="+",
        help="Level of background activity in Hz",
    )
    parser.add_argument(
        "-i_e_weight",
        type=float,
        default=[10],
        nargs="+",
        help="Weight of I-to-E synapses in mV",
    )
    parser.add_argument(
        "-e_i_weight",
        type=float,
        default=[3],
        nargs="+",
        help="Weight of E-to-I synapses in mV",
    )
    parser.add_argument(
        "-e_e_max_weight",
        type=float,
        default=[10],
        nargs="+",
        help="Maximum weight of E-to-E synapses in mV",
    )
    parser.add_argument(
        "-i_frequency",
        type=float,
        default=[20],
        nargs="+",
        help="Frequency of I input in Hz",
    )
    parser.add_argument(
        "-cue_percentage",
        type=float,
        default=[100],
        nargs="+",
        help="Percentage of neurons in the attractor to be stimulated",
    )
    parser.add_argument(
        "-cue_length",
        type=float,
        default=[1],
        nargs="+",
        help="Duration of the cue in seconds",
    )
    parser.add_argument(
        "-num_attractors",
        type=int,
        default=[3],
        nargs="+",
        help="Number of attractors to be generated",
    )
    parser.add_argument(
        "-attractor_size",
        type=int,
        default=[64],
        nargs="+",
        help="Number of neurons in each attractor",
    )
    parser.add_argument(
        "-network_size",
        type=int,
        default=[256],
        nargs="+",
        help="Number of neurons in the network",
    )
    parser.add_argument(
        "-num_cues",
        type=int,
        default=[10],
        nargs="+",
        help="Number of cues given to the network",
    )
    parser.add_argument(
        "-num_samples",
        type=int,
        default=2,
        help="Number of parameters to pick from each distribution",
    )
    parser.add_argument(
        "-coefficient_variation",
        type=float,
        default=0.2,
        help="CV when using gaussian distribution (pass one value for all parameters)",
    )
    parser.add_argument(
        "-sweep", type=str, default=[], nargs="+", help="Parameters to sweep"
    )
    parser.add_argument(
        "-cross_validation",
        type=int,
        default=2,
        help="Number of cross validation folds",
    )
    parser.add_argument("-plot", type=bool, default=False, help="Plot the results")

    args = parser.parse_args()

    tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(tmp_folder)
    print("TMP:", tmp_folder)

    num_cpus = pathos.multiprocessing.cpu_count()
    cv = args.cross_validation

    # telegram_token = "6491481149:AAFomgrhyBRohH4szH5jPT2_AoAdOYA_flY"
    telegram_token = "6488991500:AAEIZwY1f0dioEK-R8vPYMatnmmb_gCobZ8"  # Test
    telegram_token = None

    msg_args = ""
    for k, v in vars(args).items():
        msg_args += f"{k}: {v}, "
    telegram_bot = TelegramNotify(token=telegram_token)
    telegram_bot.unpin_all()
    main_msgs = telegram_bot.send_timestamped_messages(
        f"Starting RCN sweep with the following parameters: {msg_args}\ntmp_folder: {tmp_folder}"
    )

    param_grid = {
        k: v for k, v in vars(args).items() if isinstance(v, list) and k != "sweep"
    }

    # -- generate samples
    sigmas = [args.coefficient_variation * v[0] for v in param_grid.values()]
    param_samples = np.random.multivariate_normal(
        np.ravel(list(param_grid.values())),
        np.eye(len(param_grid)) * np.square(sigmas),
        args.num_samples,
    ).tolist()

    param_list_dict = []
    for p in param_samples:
        param_list_dict.append({k: v for k, v in zip(param_grid.keys(), p)})
    # -- parameters that were not explicitly set are given their default value
    for p in param_list_dict:
        for k in p.keys():
            if k not in args.sweep:
                p[k] = vars(args)[k][0]
    # param_list_dict.append({k: get_default(k) for k, v in param_grid.items()})
    param_list_dict.append({k: args.__dict__[k][0] for k, v in param_grid.items()})

    print_dict = {k: [] for k in param_list_dict[0].keys()}
    for p in param_list_dict:
        for k, v in p.items():
            print_dict[k].append(p[k])
    for k, v in print_dict.items():
        if len(np.unique(v)) == 1:
            print_dict[k] = [v[0]]
        else:
            v.sort()

    print(print_dict)

    # -- generate cv samples
    param_grid_pool = [x.copy() for x in param_list_dict for _ in range(cv)]

    # hack to keep track of process ids
    for i, g in enumerate(param_grid_pool):
        g["worker_id"] = i % num_cpus

    # chunked_param_grid = list(chunks(param_grid_pool, num_cpus))

    telegram_msgs = telegram_bot.reply_to_timestamped_messages(
        f"*0/{len(param_grid_pool)}* Waiting for first evaluation to finish.", main_msgs
    )

    results = tqdm_pathos.map(
        partial(
            run_rcn,
            show_plot=False,
            save_plot=False,
            tmp_folder=tmp_folder,
            attractor_conflict_resolution="3",
            progressbar=os.isatty(sys.stdout.fileno()),
            already_in_tmp_folder=True,
            telegram_update=(telegram_token, telegram_msgs),
        ),
        param_grid_pool,
        n_cpus=num_cpus,
    )

    score_param_names = ["f1_score", "recall", "accuracy", "triggered", "spontaneous"]

    # create a dataframe with the raw results
    df_results = pd.DataFrame(results)

    # aggregate results by the sweeped parameters
    if not args.sweep:
        sweeped_param_names = list(df_results.columns[:-5])
    else:
        sweeped_param_names = args.sweep
    df_results_aggregated = df_results.copy()
    df_results_aggregated = (
        df_results_aggregated.groupby(sweeped_param_names).mean().reset_index()
    )
    df_results_aggregated.sort_values(by=args.sweep, ascending=True, inplace=True)
    df_results_aggregated["sweeped"] = df_results_aggregated[args.sweep].apply(
        lambda x: ", ".join(x.apply("{:,.2f}".format)), axis=1
    )
    df_results_aggregated.to_csv(f"{tmp_folder}/results.csv")

    # print best results in order
    # print(
    #     df_results_aggregated[args.sweep + score_param_names].sort_values(
    #         by='f1_score',
    #         ascending=False))

    # save results folder
    save_folder = (
        f'RESULTS/SWEEP_SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
    )
    os.makedirs(save_folder)

    # plot results for cue time
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    for p in sweeped_param_names:
        axis = df_results_aggregated.plot(kind="bar", x=p, y="f1_score", legend=False)
        axis.set_ylabel("F1 score")
        axis.set_xlabel(args.sweep)
        axis.set_ylim([0, 1])
        axis.set_xticklabels(df_results_aggregated["sweeped"], rotation=45)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.set_title(p.replace("_", " ").title())
        fig = plt.gcf()
        fig.savefig(f"{save_folder}/sweep_results_{p}.png")
        fig.show()

    # Find best parameters
    # best_params = df_results_aggregated.loc[df_results_aggregated.f1_score.idxmax()]
    # best_score = best_params['f1_score']
    # best_params = best_params.drop(score_param_names + ['sweeped']).to_dict()
    # print(f'Best parameters: {best_params}')
    #
    # telegram_bot.reply_to_timestamped_messages(
    #     f"Finished RCN sweep with the following results:\n{best_params}\nScore: {best_score}", main_msgs)

    # Run model with the best parameters and plot output
    # run_rcn(best_params, tmp_folder=tmp_folder, save_plot=save_folder, low_memory=False,
    #         attractor_conflict_resolution='3')
    # os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
    os.rename(f"{tmp_folder}/results.csv", f"{save_folder}/results.csv")

    telegram_bot.reply_to_timestamped_messages(
        f"Saved results to {save_folder}", main_msgs
    )
    telegram_bot.unpin_all()

    cleanup()

    # TODO ordering of parameters in sweeped plot is missing so it makes sense to calculate performance metric and visualise that or calculate robustness to one parameter at a time
