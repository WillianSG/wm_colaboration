# -*- coding: utf-8 -*-
"""
@author: Thomas/Willian
"""

import shutil
import time
import os
import warnings
import copy

from tqdm import TqdmWarning, tqdm
from brian2 import *

from recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.other import make_timestamped_folder, compute_ps_score
from helper_functions.population_spikes import count_ps
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin

def run_rcn(
        params, tmp_folder=".", show_plot=False, save_plot=None, progressbar=True, seed_init=None, low_memory=True,
        attractor_conflict_resolution='0'):
    warnings.filterwarnings("ignore", category=TqdmWarning)

    if show_plot == True and low_memory == True:
        raise ValueError("plotting is only possible with low_memory=False")

    pid = os.getpid()
    # print(f'RUNNING worker {pid} with params: {params}')

    ba = params["background_activity"]
    i_e_w = params["i_e_weight"]
    e_i_w = params["e_i_weight"]
    e_e_maxw = params["e_e_max_weight"]
    i_freq = params["i_frequency"]
    cue_percentage = params["cue_percentage"]
    cue_time = params["cue_length"]
    num_attractors = int(params["num_attractors"])
    num_cues = int(params["num_cues"])
    attractor_size = int(params["attractor_size"])
    network_size = int(params["network_size"])
    worker_id = params["worker_id"] if "worker_id" in params else None

    # Attractor size conflict resolution
    import sympy
    a, s, n = sympy.symbols("a s n")
    expr = a * (s + 16) <= n
    if not expr.subs([(a, num_attractors), (s, attractor_size), (n, network_size)]):
        if attractor_conflict_resolution == '0':
            print(
                f"{num_attractors} attractors of size {attractor_size} cannot fit into a network of {network_size} neurons."
            )
            choice = input(
                "1- Smaller attractors\n2- Fewer attractors\n3- Larger network\n"
            )
        else:
            choice = attractor_conflict_resolution
        if choice == "1":
            asol = sympy.solveset(
                expr.subs([(a, num_attractors), (n, network_size)]), s, sympy.Integers
            )
            attractor_size = asol.sup
            # attractor_size = floor((network_size - num_attractors * 16) / num_attractors)
            print(f"Setting optimal attractor size: {attractor_size}")
        if choice == "2":
            asol = sympy.solveset(
                expr.subs([(s, attractor_size), (n, network_size)]), a, sympy.Integers
            )
            num_attractors = asol.sup
            print(f"Settin optimal number of attractors: {num_attractors}")
        if choice == "3":
            asol = sympy.solveset(
                expr.subs([(a, num_attractors), (s, attractor_size)]), n, sympy.Integers
            )
            network_size = asol.inf
            print(f"Setting optimal network size: {network_size}")

    rcn = RecurrentCompetitiveNet(
        stim_size=attractor_size,
        plasticity_rule="LR4",
        parameter_set="2.2",
        seed_init=seed_init,
        low_memory=low_memory,
    )

    plastic_syn = False
    plastic_ux = True
    rcn.E_E_syn_matrix_snapshot = False
    rcn.w_max = e_e_maxw * mV  # for param. 2.1: 10*mV
    rcn.w_e_i = e_i_w * mV  # 3 mV default
    rcn.w_i_e = i_e_w * mV  # 10 mV default
    rcn.spont_rate = ba * Hz
    rcn.N_e = network_size
    rcn.N_input_e = network_size
    rcn.N_i = network_size // 4
    rcn.N_input_i = network_size // 4

    # -- intrinsic plasticity setup (Vth_e_decr for naive / tau_Vth_e for calcium-based)
    rcn.tau_Vth_e = 0.1 * second  # 0.1 s default
    # rcn.Vth_e_init = -51.6 * mV           # -52 mV default
    rcn.k = 4  # 2 default

    rcn.net_init()

    # -- synaptic augmentation setup
    rcn.U = 0.2  # 0.2 default

    rcn.set_stimulus_i(stimulus="flat_to_I", frequency=i_freq * Hz)  # default: 15 Hz

    # --folder for simulation results
    rcn.net_sim_data_path = make_timestamped_folder(os.path.join(tmp_folder, f"{pid}"))

    # output .txt with simulation summary.
    _f = os.path.join(rcn.net_sim_data_path, "simulation_summary.txt")

    attractors_list = []
    for i in range(num_attractors):
        stim_id = rcn.set_active_E_ids(
            stimulus="flat_to_E_fixed_size", offset=i * (attractor_size + 16)
        )
        rcn.set_potentiated_synapses(stim_id)
        attractors_list.append([f"A{i}", stim_id])

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    # generate shuffled attractors
    """Attractor cueing order:
        - Attractor name
        - Neuron IDs belonging to attractor
        - Length of triggered window
        - Generic stimulus
            - % of neurons targeted within attractor
            - IDs of targeted neurons
            - (start time of GS, end time of GS)"""
    attractors_cueing_order = [copy.deepcopy(random.choice(attractors_list))]
    for i in range(num_cues - 1):
        atr_sel = copy.deepcopy(random.choice(attractors_list))
        while attractors_cueing_order[i][0] == atr_sel[0]:
            atr_sel = copy.deepcopy(random.choice(attractors_list))
        attractors_cueing_order.append(atr_sel)

    # generate random reactivation times in range
    for a in attractors_cueing_order:
        a.append(random.uniform(1, 5))

    overall_sim_time = len(attractors_cueing_order) * cue_time + sum(
        [a[2] for a in attractors_cueing_order]
    )
    pbar = tqdm(
        total=overall_sim_time,
        disable=not progressbar,
        unit="sim s",
        leave=True,
        position=worker_id,
        desc=f"WORKER {worker_id}",
        bar_format="{l_bar}{bar}|{n:.2f}/{total:.2f} [{elapsed}<{remaining},{rate_fmt}{postfix}]",
    )
    pbar.set_description(f"WORKER {worker_id} with params {params}")
    time.sleep(0.1)
    pbar.update(0)
    for a in attractors_cueing_order:
        gs = (cue_percentage, a[1], (rcn.net.t / second, rcn.net.t / second + cue_time))
        act_ids = rcn.generic_stimulus(
            frequency=rcn.stim_freq_e, stim_perc=gs[0], subset=gs[1]
        )
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

    trig, spont, accuracy, recall, f1_score = compute_ps_score(
        atr_ps_counts, attractors_cueing_order
    )

    if show_plot or save_plot is not None:
        title_addition = (
            f"BA {ba} Hz, GS {cue_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz"
        )
        filename_addition = f"_BA_{ba}_GS_{cue_percentage}_W_{i_e_w}_Hz_{i_freq}"

        fig = plot_x_u_spks_from_basin(
            path=rcn.net_sim_data_path,
            filename="x_u_spks_from_basin" + filename_addition,
            title_addition=title_addition,
            rcn=rcn,
            attractor_cues=attractors_cueing_order,
            pss_categorised=atr_ps_counts,
            num_neurons=len(rcn.E),
            show=show_plot,
        )

        if save_plot is not None:
            fig.savefig(os.path.join(save_plot, f"x_u_spks_from_basin{filename_addition}.png"))

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

    params.pop("worker_id", None)

    return params | {
        "f1_score": f1_score,
        "recall": recall,
        "accuracy": accuracy,
        "triggered": trig,
        "spontaneous": spont,
    }