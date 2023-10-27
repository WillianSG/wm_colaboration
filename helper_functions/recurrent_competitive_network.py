# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl / w.soares.girao@rug.nl
@university: University of Groningen
@group: CogniGron
"""

import glob
import pickle
import shutil
import sys
import time
from time import localtime, strftime
import os
import warnings
import copy

from brian2 import (
    prefs,
    Hz,
    second,
    seed,
    Equations,
    NeuronGroup,
    Synapses,
    Quantity,
    SpikeMonitor,
    StateMonitor,
    PopulationRateMonitor,
    Clock,
    defaultclock,
    Network,
    network_operation,
)
from tqdm import TqdmWarning, tqdm

from helper_functions.other import make_timestamped_folder, compute_ps_score
from helper_functions.population_spikes import count_ps
from helper_functions.telegram_notify import TelegramNotify
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin

prefs.codegen.target = "numpy"


def is_pycharm():
    return os.getenv("PYCHARM_HOSTED") != None


if is_pycharm():
    from helper_functions.load_rule_parameters import *
    from helper_functions.load_synapse_model import *
    from helper_functions.load_stimulus import *
else:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__, "../")))
    sys.path.append(os.path.join(root, "helper_functions"))
    from load_rule_parameters import *
    from load_synapse_model import *
    from load_stimulus import *


def count_spikes():
    pass


def compute_energy():
    pass


def run_rcn(params, tmp_folder=".", show_plot=False, save_plot=None, progressbar=True, seed_init=None, low_memory=True,
            attractor_conflict_resolution='0', already_in_tmp_folder=False, telegram_update=None,
            return_complete_stats=False):
    warnings.filterwarnings("ignore", category=TqdmWarning)

    if show_plot == True and low_memory == True:
        raise ValueError("plotting is only possible with low_memory=False")
    if save_plot is False:
        save_plot = None

    pid = os.getpid()
    # print(f'RUNNING worker {pid} with params: {params}')

    ba = params["background_activity"]
    i_e_w = params["i_e_weight"]
    e_i_w = params["e_i_weight"]
    e_e_maxw = params["e_e_max_weight"]
    i_freq = params["i_frequency"]
    cue_percentage = params["cue_percentage"]
    cue_time = params["cue_length"]
    recall_time = params["recall_time"] if "recall_time" in params else (1, 5)
    if not isinstance(recall_time, tuple):
        recall_time = (recall_time, recall_time)
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
    if already_in_tmp_folder:
        rcn.net_sim_data_path = make_timestamped_folder('.', addition=f'_{pid}')
    else:
        rcn.net_sim_data_path = make_timestamped_folder(tmp_folder, addition=f'_{pid}')

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
        if not len(attractors_list) == 1:
            while attractors_cueing_order[i][0] == atr_sel[0]:
                atr_sel = copy.deepcopy(random.choice(attractors_list))
        attractors_cueing_order.append(atr_sel)

    # generate random reactivation times in range
    for a in attractors_cueing_order:
        a.append(random.uniform(*recall_time))

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
            f"BA {ba:.2f} Hz, E-E max {e_e_maxw:.2f} mV, E-I {e_i_w:.2f} mV, I-E {i_e_w:.2f} mV, I {i_freq:.2f} Hz"
        )

        fig = plot_x_u_spks_from_basin(
            title_addition=title_addition,
            rcn=rcn,
            attractor_cues=attractors_cueing_order,
            pss_categorised=atr_ps_counts,
            num_neurons=len(rcn.E),
        )

        if show_plot:
            fig.show()

        if save_plot is not None:
            fig.savefig(os.path.join(save_plot, f"ATR_plot.svg"))

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
    # if low_memory:
    #     del rcn
    pbar.close()

    params.pop("worker_id", None)

    if telegram_update is not None:
        telegram_bot = TelegramNotify(token=telegram_update[0])
        # hack to keep track of last update
        telegram_bot.read_pinned_and_increment_it(telegram_update[1], f1_score)

    if return_complete_stats:
        return rcn, params | {
            "f1_score": f1_score,
            "recall": recall,
            "accuracy": accuracy,
            "triggered": trig,
            "spontaneous": spont,
            "ps_counts": atr_ps_counts,
            "attractors_cueing_order": attractors_cueing_order
        }
    else:
        return params | {
            "f1_score": f1_score,
            "recall": recall,
            "accuracy": accuracy,
            "triggered": trig,
            "spontaneous": spont,
        }


class RecurrentCompetitiveNet:
    def __init__(
            self,
            stim_size=64,
            plasticity_rule="LR2",
            parameter_set="2.4",
            t_run=2 * second,
            seed_init=None,
            low_memory=False,
            sFSA=False
    ):

        self.sFSA = sFSA

        seed(seed_init)

        # When True then only E_mon is instantiated as it's the only one necessary to compute the PS score
        # When plotting the network activity, set this to False
        self.low_memory = low_memory

        # ------ state transition connections
        self.p_A2GO = 0.15
        self.delay_A2GO = 2 * second

        self.thr_GO_state = -48.5

        self.p_A2B = 0.15
        self.delay_A2B = 0.525 * second

        # ------ simulation parameters
        self.net_id = strftime("%d%b%Y_%H-%M-%S", localtime())

        self.net_sim_path = os.path.dirname(
            os.path.abspath(os.path.join(__file__, "../"))
        )

        self.net_sim_data_path = ""

        self.t_run = t_run
        self.dt = 0.1 * ms

        self.tqdm_bar = None

        self.int_meth_neur = "euler"
        self.int_meth_syn = "euler"

        # ------ network parameters
        self.stimulus_pulse = False
        # self.stimulus_pulse_duration = self.t_run - 1 * second
        self.stimulus_pulse_clock_dt = 0.1 * ms

        self.stim_size_e = stim_size
        self.stim_size_i = 64

        # ------ neurons parameters
        self.neuron_type = "LIF"

        self.N_input_e = 256  # num. of input neurons for E
        self.N_input_i = 64  # num. of input neurons for I

        # excitatory population

        self.stim_freq_e = 6600 * Hz
        self.stim_freq_i = 3900 * Hz
        self.spont_rate = 10 * Hz

        self.N_e = 256  # num. of neurons
        self.Vr_e = -65 * mV  # resting potential
        self.Vrst_e = -65 * mV  # reset potential
        self.Vth_e_init = -45 * mV  # initial threshold voltage
        self.Vth_e_decr = 5 * mV  # post-spike threshold voltage increase
        self.tau_Vth_e = 20 * ms  # time constant of threshold decay
        self.taum_e = 20 * ms  # membrane time constant
        self.tref_e = 2 * ms  # refractory period
        self.tau_epsp_e = 3.5 * ms  # time constant of EPSP
        self.tau_ipsp_e = 5.5 * ms  # time constant of IPSP

        self.k = 2  # shape of f(u) function for calcium-threshold

        # inhibitory population

        self.N_i = 64  # num. of neurons
        self.Vr_i = -60 * mV  # resting voltage
        self.Vrst_i = -60 * mV  # reset potential
        self.Vth_i = -40 * mV  # threshold voltage
        self.taum_i = 10 * ms  # membrane time constant
        self.tref_i = 1 * ms  # refractory period
        self.tau_epsp_i = 3.5 * ms  # time constant of EPSP
        self.tau_ipsp_i = 5.5 * ms  # time constant of IPSP

        # ------ synapse parameters
        # plasticity rule
        self.plasticity_rule = plasticity_rule
        self.rule_parameters = parameter_set
        self.bistability = True
        self.stop_learning = False
        self.resources = True

        # connection probabilities
        self.p_e_i = 0.25  # excitatory to inhibitory (0.25)
        self.p_i_e = 0.25  # inhibitory to excitatory (0.25)
        self.p_e_e = 0.4  # excitatory to excitatory (0.4)
        self.p_i_i = 0.0  # inhibitory to inhibitory (0.0)

        # delays
        self.syn_delay_Vepsp_e_e = 3 * ms
        self.syn_delay_w_update_e_e = 0 * ms
        self.E_I_delay = 0 * ms

        # weights
        self.w_e_e_max = 7.5 * mV
        self.w_i_i_max = 7.5 * mV

        self.w_input_e = 1 * mV
        self.w_input_e_spont = 70 * mV
        self.w_input_i = 60 * mV
        self.w_e_i = 1 * mV
        self.w_i_e = 1 * mV
        # self.w_e_e = 0.5 * mV
        # self.w_i_i = 0.5 * mV

        # data save
        self.M_ee = []

        self.E_E_syn_matrix_snapshot = False
        self.E_E_syn_matrix_snapshot_dt = 100 * ms
        self.E_E_syn_matrix_path = ""

        # ------ data (monitors) parameters
        self.rec_dt = 1 * ms

        self.Input_to_E_mon_record = True
        self.Input_to_I_mon_record = True
        self.E_mon_record = True
        self.I_mon_record = True
        self.Input_E_rec_record = True
        self.Input_I_rec_record = True
        self.E_rec_record = True
        self.I_rec_record = True
        self.E_E_rec_record = True
        self.E_I_rec_record = True
        self.I_E_rec_record = True

        self.Input_E_rec_attributes = "w"
        self.Input_I_rec_attributes = "w"
        self.E_rec_attributes = ["Vm", "Vth_e"]
        self.I_rec_attributes = ("Vm")
        self.E_E_rec_attributes = ["w"]
        self.E_I_rec_attributes = "w"
        self.I_E_rec_attributes = "w"

        if self.plasticity_rule == "LR4":
            self.E_E_rec_attributes.append("x_")
            self.E_rec_attributes.append("u")

        # ------ misc operation variables
        self.stimulus_neurons_e_ids = []
        self.stimulus_neurons_i_ids = []

        # to cache the values loaded from dumped monitors using the load_monitors_from_file method
        self.dumped_mons_dict = {}
        # to cache PySpike objects
        self.pyspike_spks = []
        # to cache pss results
        self.x_pss = {}
        self.y_pss = {}
        self.y_smooth_pss = {}
        self.pss = {}

    # 1 ------ setters

    # 1.1 ------ neurons

    """
    Sets the objects representing the excitatory (E), inhibitory (I) and input neuron populations used in the model.
    - LIF model with adaptive threshold.
    """

    def set_neuron_pop(self):
        # equations (voltage based neuron model)
        # Gompertz:
        # a = 1
        # b = 2.7083893552094156
        # c = 5.509734056519429
        self.eqs_e = Equations(
            """
            plastic_u : boolean (shared)
            dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / taum_e : volt (unless refractory)
            dVepsp/dt = -Vepsp / tau_epsp : volt
            dVipsp/dt = -Vipsp / tau_ipsp : volt
            du/dt = ((U - u) / tau_f) * int(plastic_u) : 1
            dVth_e/dt = (Vth_e_init - (0.004 * (3 * exp(-exp(3.4 - 9.5 * u))) * volt) - Vth_e) / tau_Vth_e  : volt
            """,
            Vr_e=self.Vr_e,
            taum_e=self.taum_e,
            tau_epsp=self.tau_epsp_e,
            tau_ipsp=self.tau_ipsp_e,
            Vth_e_init=self.Vth_e_init,
            tau_Vth_e=self.tau_Vth_e,
        )

        # self.eqs_e = Equations('''
        #     plastic_u : boolean (shared)
        #     dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / taum_e : volt (unless refractory)
        #     dVepsp/dt = -Vepsp / tau_epsp : volt
        #     dVipsp/dt = -Vipsp / tau_ipsp : volt
        #     du/dt = ((U - u) / tau_f) * int(plastic_u) : 1
        #     dVth_e/dt = (Vth_e_init - (0.02*(1/(1+exp(u*6))) * volt) - Vth_e) / tau_Vth_e  : volt
        #     ''',
        #                        Vr_e=self.Vr_e,
        #                        taum_e=self.taum_e,
        #                        tau_epsp=self.tau_epsp_e,
        #                        tau_ipsp=self.tau_ipsp_e,
        #                        Vth_e_init=self.Vth_e_init,
        #                        tau_Vth_e=self.tau_Vth_e)

        self.eqs_i = Equations(
            """
            dVm/dt = (Vepsp - Vipsp - (Vm - Vr_i)) / taum_i : volt (unless refractory)
            dVepsp/dt = -Vepsp / tau_epsp : volt
            dVipsp/dt = -Vipsp / tau_ipsp : volt""",
            Vr_i=self.Vr_i,
            taum_i=self.taum_i,
            tau_epsp=self.tau_epsp_i,
            tau_ipsp=self.tau_ipsp_i,
        )

        # populations
        self.E = NeuronGroup(
            N=self.N_e,
            model=self.eqs_e,
            reset="""
                                    Vm = Vrst_e
                                    u = u + U * (1 - u) * int(plastic_u)
                                    """,
            threshold="Vm > Vth_e",
            refractory=self.tref_e,
            method=self.int_meth_neur,
            name="E",
        )

        self.I = NeuronGroup(
            N=self.N_i,
            model=self.eqs_i,
            reset="Vm = Vrst_i",
            threshold="Vm > Vth_i",
            refractory=self.tref_i,
            method=self.int_meth_neur,
            name="I",
        )

        self.Input_to_E = NeuronGroup(
            N=self.N_input_e,
            model="rates : Hz",
            threshold="rand()<rates*dt",
            name="Input_to_E",
        )

        self.Input_to_I = NeuronGroup(
            N=self.N_input_i,
            model="rates : Hz",
            threshold="rand()<rates*dt",
            name="Input_to_I",
        )

        self.Input_to_E_spont = NeuronGroup(
            N=self.N_input_e,
            model="rates : Hz",
            threshold="rand()<rates*dt",
            name="Input_to_E_spont",
        )

        # populations's attributes
        self.E.Vth_e = self.Vth_e_init

        # rand init membrane voltages
        # self.E.Vm = (self.Vrst_e + rand(self.N_e) * (self.Vth_e_init - self.Vr_e))
        # self.I.Vm = (self.Vrst_i + rand(self.N_i) * (self.Vth_i - self.Vr_i))

        self.E.Vm = self.Vrst_e
        self.I.Vm = self.Vrst_i

    """
    Sets the ids of the active neurons on the input before actually loading the stimulus.
    """

    def set_active_E_ids(self, stimulus, offset=0):
        stim_ids = load_stimulus(
            stimulus_type=stimulus, stimulus_size=self.stim_size_e, offset=offset
        )
        self.stimulus_neurons_e_ids = np.append(
            self.stimulus_neurons_e_ids, stim_ids
        ).astype(int)

        return stim_ids

        # 1.2 ------ synapses

    """
    Loads synaptic rule equations.
    """

    def set_learning_rule(self):
        # rule's equations
        [self.model_E_E, self.pre_E_E, self.post_E_E] = load_synapse_model(
            plasticity_rule=self.plasticity_rule,
            neuron_type=self.neuron_type,
            bistability=self.bistability,
            stoplearning=self.stop_learning,
            resources=self.resources,
        )

        # rule's parameters
        [
            self.tau_xpre,
            self.tau_xpost,
            self.xpre_jump,
            self.xpost_jump,
            self.rho_neg,
            self.rho_neg2,
            self.rho_init,
            self.tau_rho,
            self.thr_post,
            self.thr_pre,
            self.thr_b_rho,
            self.rho_min,
            self.rho_max,
            self.alpha,
            self.beta,
            self.xpre_factor,
            self.w_max,
            self.xpre_min,
            self.xpost_min,
            self.xpost_max,
            self.xpre_max,
            self.tau_xstop,
            self.xstop_jump,
            self.xstop_max,
            self.xstop_min,
            self.thr_stop_h,
            self.thr_stop_l,
            self.U,
            self.tau_d,
            self.tau_f,
        ] = load_rule_parameters(self.plasticity_rule, self.rule_parameters)

    """
    Sets synapses objects and connections.
    """

    def set_synapses(self):
        # creating synapse instances
        self.Input_E = Synapses(
            source=self.Input_to_E,
            target=self.E,
            model='w : volt',
            on_pre='Vepsp += w',
            name='Input_E')

        self.Input_E_spont = Synapses(
            source=self.Input_to_E_spont,
            target=self.E,
            model='w : volt',
            on_pre='Vepsp += w',
            name='Input_E_spont')

        self.Input_I = Synapses(
            source=self.Input_to_I,
            target=self.I,
            model='w : volt',
            on_pre='Vepsp += w',
            name='Input_I')

        self.E_I = Synapses(
            source=self.E,
            target=self.I,
            model='w : volt',
            on_pre='Vepsp += w',
            delay=self.E_I_delay,
            name='E_I')

        self.I_E = Synapses(
            source=self.I,
            target=self.E,
            model='w : volt',
            on_pre='Vipsp += w',
            name='I_E')

        self.E_E = Synapses(  # E-E plastic synapses
            source=self.E,
            target=self.E,
            model=self.model_E_E,
            on_pre=self.pre_E_E,
            on_post=self.post_E_E,
            method=self.int_meth_syn,
            name='E_E')

        # f'''w_ef = w*int(Vth_e_post < {self.thr_GO_state}*mV)*int(Vth_e_pre < {self.thr_GO_state}*mV), Vepsp += w_ef'''
        self.A_2_B_synapses = Synapses(  # synapses between different attractors
            source=self.E,
            target=self.E,
            model='''w_ef : volt
            w : volt''',
            on_pre=f'''w_ef = w*int(Vth_e_post < {self.thr_GO_state}*mV)*int(Vth_e_pre < {self.thr_GO_state}*mV)
            Vepsp += w_ef''',
            delay=self.delay_A2B,
            name='A_2_B_synapses')

        self.A_2_GO_synapses = Synapses(
            source=self.E,
            target=self.E,
            model='''w_ef : volt
            w : volt''',
            on_pre=f'''w_ef = w
            Vepsp += w_ef''',
            delay=self.delay_A2GO,
            name='A_2_GO_synapses')

        # connecting synapses
        self.Input_E.connect(j='i')
        self.Input_E_spont.connect(j='i')
        self.Input_I.connect(j='i')

        self.E_I.connect(True, p=self.p_e_i)
        self.I_E.connect(True, p=self.p_i_e)
        self.E_E.connect('i!=j', p=self.p_e_e)
        self.A_2_B_synapses.connect('i!=j', p=self.p_A2B)
        self.A_2_GO_synapses.connect('i!=j', p=self.p_A2GO)

        # init synaptic variables
        self.Input_E.w = self.w_input_e
        self.Input_E_spont.w = self.w_input_e_spont
        self.Input_I.w = self.w_input_i
        self.E_I.w = self.w_e_i
        self.I_E.w = self.w_i_e

        self.A_2_B_synapses.w = 0 * mV
        self.A_2_GO_synapses.w = 0 * mV

        if self.plasticity_rule == 'LR4':
            self.E_E.x_ = 1.0
            self.E_E.u = self.U

        self.E_E.Vepsp_transmission.delay = self.syn_delay_Vepsp_e_e

    """
    Allows weight state variables change in synaptic model.
    """

    def set_E_E_plastic(self, plastic=False):
        self.E_E.plastic = plastic

    """
    Allows resources (x) and utilization (u) state variables change in synaptic model.
    """

    def set_E_E_ux_vars_plastic(self, plastic=False):
        self.E_E.plastic_x = plastic
        self.E.plastic_u = plastic

    """
    Creates matrix spikemon_P from E to E connections.
    """

    def set_E_syn_matrix(self):
        self.M_ee = np.full((len(self.E), len(self.E)), np.nan)
        self.M_ee[self.E_E.i[:], self.E_E.j[:]] = self.E_E.rho[:]

    """
    """

    def set_random_E_E_syn_w(self, percent=0.5):
        self.set_E_syn_matrix()

        for pre_id in range(0, len(self.E)):
            for post_id in range(0, len(self.E)):
                if np.isnan(self.M_ee[pre_id][post_id]) == False:
                    percent_ = np.uniform(0, 1)
                    if percent_ < percent:
                        s = np.uniform(0, 1)
                        self.E_E.rho[pre_id, post_id] = round(s, 2)
                    else:
                        self.E_E.rho[pre_id, post_id] = 0.0

    """
    Set synapses between neurons receiving input potentiated.
    Warning: method 'set_active_E_ids()' has to be called before the call of this function in case the stimulus to be
    provided to the network isn't set yet.
    """

    def set_potentiated_synapses(self, stim_ids):
        for x in range(0, len(self.E_E)):
            if self.E_E.i[x] in stim_ids and self.E_E.j[x] in stim_ids:
                self.E_E.rho[self.E_E.i[x], self.E_E.j[x]] = 1.0
                self.E_E.w[self.E_E.i[x], self.E_E.j[x]] = self.w_max

    def set_synapses_A_2_B(self, A_ids, B_ids, weight):
        '''
            Connections implementing 'cue transfer' of the sFSM (A,i)->B mapping: this
        method configures connections from i to A to implement a 'cue transfer'.

        Obs.: The connection delay between i and A should be similart to the cueing time of i.
        '''
        for x in range(0, len(self.A_2_B_synapses)):
            if self.A_2_B_synapses.i[x] in A_ids and self.A_2_B_synapses.j[x] in B_ids:
                self.A_2_B_synapses.w[self.A_2_B_synapses.i[x], self.A_2_B_synapses.j[x]] = weight

    def set_synapses_A_2_GO(self, A_ids, GO_ids, weight):
        '''
            Connections implementing the (A,i)->B mapping: this method configures connections from i
            to B and from A to B.

        Obs.: The connection delay should be around twice the cueing time of i.
        '''
        for x in range(0, len(self.A_2_GO_synapses)):
            if self.A_2_GO_synapses.i[x] in A_ids and self.A_2_GO_synapses.j[x] in GO_ids:
                self.A_2_GO_synapses.w[self.A_2_GO_synapses.i[x], self.A_2_GO_synapses.j[x]] = weight

    # 1.4 ------ network operation

    """
    """

    def run_net(
            self,
            duration=3 * second,
            gather_every=0 * second,
            pulse_ending=False,
            callback=None,
    ):

        if sys.platform == "linux":

            pass

        else:

            pass

        if not isinstance(duration, Quantity):
            duration *= second
        if not isinstance(gather_every, Quantity):
            gather_every *= second
        if not isinstance(pulse_ending, Quantity):
            pulse_ending *= second

        if not callback:
            callback = []

        self.t_run = duration

        if gather_every == 0 * second:
            num_runs = 1
            duration = duration
        else:
            num_runs = int(round(duration / gather_every))
            duration = gather_every

        # if not self.tqdm_bar:
        #     self.tqdm_bar = tqdm(total=(self.net.t + duration) / second, desc='RCN', unit='sim s',
        #                          bar_format='{n:.1f}/{total:.1f} sim s in {elapsed} s, {rate_fmt} {postfix}]',
        #                          leave=False, dynamic_ncols=True)
        for i in range(num_runs):
            # self.tqdm_bar.total = (self.net.t + duration) / second
            # self.tqdm_bar.update(duration / second)

            # self.tqdm_bar.set_description(
            #     f'Running RCN in [{self.net.t:.1f}-{self.net.t + duration:.1f}] s, '
            #     # f'input ending at {self.stimulus_pulse_duration:.1f} s'
            # )
            self.net.run(
                duration=duration, report=None, namespace=self.set_net_namespace()
            )

            # if gather_every > 0 * second:
            #     for f in callback:
            #         f(self)

            self.net.stop()

            # self.tqdm_bar.update( duration / second )
        # self.tqdm_bar.close()

    """
    """

    def set_stimulus_e(self, stimulus, frequency, offset=0, stimulus_size=0):
        if stimulus_size == 0:
            stimulus_size = self.stim_size_e

        if stimulus != "":
            self.stimulus_neurons_e_ids = load_stimulus(
                stimulus_type=stimulus, stimulus_size=stimulus_size, offset=offset
            )

            self.Input_to_E.rates[self.stimulus_neurons_e_ids] = frequency
        else:
            self.Input_to_E.rates[self.stimulus_neurons_e_ids] = frequency

    """
    """

    def set_stimulus_i(self, stimulus, frequency, offset=0):
        if stimulus != "":
            self.stimulus_neurons_i_ids = load_stimulus(
                stimulus_type=stimulus, stimulus_size=self.stim_size_i, offset=offset
            )

            self.Input_to_I.rates[self.stimulus_neurons_i_ids] = frequency
        else:
            self.Input_to_I.rates[self.stimulus_neurons_i_ids] = frequency

    """
    Stimulates % of excitatory neurons as in Mongillo.
    """

    def generic_stimulus(self, frequency, stim_perc, subset):
        subset = np.random.choice(
            subset, int((len(subset) * stim_perc) / 100), replace=False
        )

        self.Input_to_E.rates[subset] = frequency

        return subset

    def generic_stimulus_off(self, act_ids):
        self.Input_to_E.rates[act_ids] = 0 * Hz

    """
    Stimulates 'stim_perc'% of neurons of pattern 'pattern'.
    """

    def stimulate_attractors(
            self, stimulus, frequency, stim_perc=10, offset=0, stimulus_size=0
    ):
        if stimulus_size == 0:
            stimulus_size = self.stim_size_e

        n_act_ids = int((stimulus_size * stim_perc) / 100)

        act_ids = np.random.choice(
            load_stimulus(
                stimulus_type=stimulus, stimulus_size=self.stim_size_e, offset=offset
            ),
            size=n_act_ids,
            replace=False,
        )

        self.Input_to_E.rates[act_ids] = frequency

        # 1.3 ------ network initializers

        """
        Sets and configures the rcn network objects for simulation.
        """

    def net_init(self):
        # self.set_results_folder()  # sim. results

        if not self.sFSA:  # sFSA class calls it after RCN configuration.
            self.set_learning_rule()  # rule eqs./params.

        self.set_neuron_pop()  # neuron populations
        self.set_synapses()  # syn. connections
        self.set_spk_monitors()
        self.set_state_monitors()

        self.set_net_obj()

        self.Input_to_E_spont.rates[[x for x in range(0, self.N_e)]] = self.spont_rate

    """
    Creates a folder for simulation results in the root directory.
    """

    def set_results_folder(self):
        network_sim_dir = os.path.join(self.net_sim_path, "network_results")

        if not (os.path.isdir(network_sim_dir)):
            os.mkdir(network_sim_dir)

        network_sim_dir = os.path.join(
            network_sim_dir,
            self.net_id + f"_RCN_{self.plasticity_rule}-{self.rule_parameters}",
        )

        if not (os.path.isdir(network_sim_dir)):
            os.mkdir(network_sim_dir)

        self.net_sim_data_path = network_sim_dir

        if self.E_E_syn_matrix_snapshot:
            E_E_syn_matrix_path = os.path.join(self.net_sim_data_path, "E_E_syn_matrix")

            if not (os.path.isdir(E_E_syn_matrix_path)):
                os.mkdir(E_E_syn_matrix_path)

            self.E_E_syn_matrix_path = E_E_syn_matrix_path

    """
    """

    def dump_monitors_to_file(self, monitors=None):
        # -- incrementally store all monitors to their own file
        for mon in self.spk_monitors + self.state_monitors:
            new_data = mon.get_states()
            old_data = None

            filename = f"{self.net_sim_data_path}/{mon.name}.data"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    old_data = pickle.load(f)

            if old_data is not None:
                if isinstance(mon, SpikeMonitor):
                    data_points = len(old_data["i"]) + len(new_data["i"])
                    for k, v in old_data.items():
                        if k == "i" or k == "t":
                            new_data[k] = np.hstack((old_data[k], new_data[k]))
                        elif k == "count":
                            new_data[k] = old_data[k] + new_data[k]
                        elif k == "N":
                            new_data[k] = data_points
                # elif isinstance(mon, PopulationRateMonitor):
                #     data_points = len(old_data['t']) + len(new_data['t'])
                #     for k, v in old_data.items():
                #         if k == 'rate' or k == 't':
                #             new_data[k] = np.hstack((old_data[k], v))
                #         elif k == 'N':
                #             new_data[k] = data_points
                elif isinstance(mon, StateMonitor):
                    data_points = len(old_data["t"]) + len(new_data["t"])
                    for k, v in old_data.items():
                        if k == "t":
                            new_data[k] = np.hstack((old_data[k], new_data[k]))
                        elif k == "N":
                            new_data[k] = data_points
                        else:
                            new_data[k] = np.vstack((old_data[k], new_data[k]))

            with open(filename, "wb+") as f:
                pickle.dump(new_data, f)

        # -- also store traces now, as it's really hard to extract them later
        x_traces_old = None
        if os.path.exists(f"{self.net_sim_data_path}/x_traces.data"):
            with open(f"{self.net_sim_data_path}/x_traces.data", "rb") as f:
                x_traces_old = pickle.load(f)
        x_traces_new = self.get_x_traces_from_pattern_neurons()
        # append new traces to old ones
        if x_traces_old is not None:
            for el_old in x_traces_old[0].keys():
                if el_old in x_traces_new[0]:
                    x_traces_new[0][el_old] = np.hstack(
                        (x_traces_old[0][el_old], x_traces_new[0][el_old])
                    )
            x_traces_new[1] = np.hstack((x_traces_old[1], x_traces_new[1]))
        with open(f"{self.net_sim_data_path}/x_traces.data", "wb+") as f:
            pickle.dump(x_traces_new, f)

        u_traces_old = None
        if os.path.exists(f"{self.net_sim_data_path}/u_traces.data"):
            with open(f"{self.net_sim_data_path}/u_traces.data", "rb") as f:
                u_traces_old = pickle.load(f)
        u_traces_new = self.get_u_traces_from_pattern_neurons()
        # append new traces to old ones
        if u_traces_old is not None:
            for el_old in u_traces_old[0].keys():
                if el_old in u_traces_new[0]:
                    u_traces_new[0][el_old] = np.hstack(
                        (u_traces_old[0][el_old], u_traces_new[0][el_old])
                    )
            u_traces_new[1] = np.hstack((u_traces_old[1], u_traces_new[1]))
        with open(f"{self.net_sim_data_path}/u_traces.data", "wb+") as f:
            pickle.dump(u_traces_new, f)

        self.net.remove(self.spk_monitors)
        self.net.add(self.set_spk_monitors())
        self.net.remove(self.state_monitors)
        self.net.add(self.set_state_monitors())

    def load_monitors_from_file(self):
        for f in glob.glob(os.path.join(self.net_sim_data_path, "*.data")):
            with open(f, "rb") as fp:
                self.dumped_mons_dict[
                    os.path.splitext(os.path.split(f)[1])[0]
                ] = pickle.load(fp)

        return self.dumped_mons_dict

    def set_spk_monitors(self):
        self.Input_to_E_mon = SpikeMonitor(
            source=self.Input_to_E,
            record=self.Input_to_E_mon_record,
            name="Input_to_E_mon",
        )

        self.Input_to_I_mon = SpikeMonitor(
            source=self.Input_to_I,
            record=self.Input_to_I_mon_record,
            name="Input_to_I_mon",
        )

        self.E_mon = SpikeMonitor(source=self.E, record=self.E_mon_record, name="E_mon")

        self.I_mon = SpikeMonitor(source=self.I, record=self.I_mon_record, name="I_mon")

        self.E_rate_mon = PopulationRateMonitor(source=self.E, name="E_rate_mon")

        if self.low_memory:
            self.spk_monitors = [self.E_mon]
        else:
            self.spk_monitors = [
                self.Input_to_E_mon,
                self.Input_to_I_mon,
                self.E_mon,
                self.I_mon,
            ]

        return self.spk_monitors

    """
    """

    def set_state_monitors(self):
        self.Input_E_rec = StateMonitor(
            source=self.Input_E,
            variables=self.Input_E_rec_attributes,
            record=self.Input_E_rec_record,
            dt=self.rec_dt,
            name="Input_E_rec",
        )

        self.Input_I_rec = StateMonitor(
            source=self.Input_I,
            variables=self.Input_I_rec_attributes,
            record=self.Input_I_rec_record,
            dt=self.rec_dt,
            name="Input_I_rec",
        )

        self.E_rec = StateMonitor(
            source=self.E,
            variables=self.E_rec_attributes,
            record=self.E_rec_record,
            dt=self.rec_dt,
            name="E_rec",
        )

        self.I_rec = StateMonitor(
            source=self.I,
            variables=self.I_rec_attributes,
            record=self.I_rec_record,
            dt=self.rec_dt,
            name="I_rec",
        )

        self.E_E_rec = StateMonitor(
            source=self.E_E,
            variables=self.E_E_rec_attributes,
            record=self.E_E_rec_record,
            dt=self.rec_dt,
            name="E_E_rec",
        )

        self.E_I_rec = StateMonitor(
            source=self.E_I,
            variables=self.E_I_rec_attributes,
            record=self.E_I_rec_record,
            dt=self.rec_dt,
            name="E_I_rec",
        )

        self.I_E_rec = StateMonitor(
            source=self.I_E,
            variables=self.I_E_rec_attributes,
            record=self.I_E_rec_record,
            dt=self.rec_dt,
            name="I_E_rec",
        )

        if self.low_memory:
            self.state_monitors = []
        else:
            self.state_monitors = [
                self.Input_E_rec,
                self.Input_I_rec,
                self.E_rec,
                self.I_rec,
                self.E_E_rec,
                self.E_I_rec,
                self.I_E_rec,
            ]

        return self.state_monitors

    """
    Creates a brian2 network object with the neuron/synapse objects defined.
    """

    def set_net_obj(self):
        self.stimulus_pulse_clock = Clock(
            self.stimulus_pulse_clock_dt, name="stim_pulse_clk"
        )

        self.E_E_syn_matrix_clock = Clock(
            self.E_E_syn_matrix_snapshot_dt, name="E_E_syn_matrix_clk"
        )

        # deactivate stimulus
        # if self.stimulus_pulse:
        # @network_operation( clock=self.stimulus_pulse_clock )
        # def stimulus_pulse():
        # if defaultclock.t >= self.stimulus_pulse_duration:
        #     self.set_stimulus_e( stimulus='', frequency=0 * Hz )
        # else:
        #     @network_operation( clock=self.stimulus_pulse_clock )
        #     def stimulus_pulse():
        #         pass
        #
        if self.E_E_syn_matrix_snapshot:

            @network_operation(clock=self.E_E_syn_matrix_clock)
            def store_E_E_syn_matrix_snapshot():
                synaptic_matrix = np.full((len(self.E), len(self.E)), -1.0)
                synaptic_matrix[self.E_E.i, self.E_E.j] = self.E_E.rho

                fn = os.path.join(
                    self.E_E_syn_matrix_path,
                    str(int(round(defaultclock.x / ms, 0)))
                    + "_ms_E_E_syn_matrix.pickle",
                )

                with open(fn, "wb") as f:
                    pickle.dump((synaptic_matrix), f)

        else:

            @network_operation(clock=self.E_E_syn_matrix_clock)
            def store_E_E_syn_matrix_snapshot():
                pass

        defaultclock.dt = self.dt

        self.net = Network(
            self.Input_to_E,
            self.Input_to_E_spont,
            self.Input_to_I,
            self.E,
            self.I,
            self.Input_E,
            self.Input_E_spont,
            self.Input_I,
            self.E_I,
            self.I_E,
            self.E_E,
            self.A_2_B_synapses,
            self.A_2_GO_synapses,
            self.spk_monitors,
            self.state_monitors,
            store_E_E_syn_matrix_snapshot,
            name="rcn_net",
        )

    """
    """

    def set_net_namespace(self):
        self.namespace = {
            "Vrst_e": self.Vrst_e,
            "Vth_e_init": self.Vth_e_init,
            "Vrst_i": self.Vrst_i,
            "Vth_i": self.Vth_i,
            "Vth_e_decr": self.Vth_e_decr,
            "tau_xpre": self.tau_xpre,
            "tau_xpost": self.tau_xpost,
            "xpre_jump": self.xpre_jump,
            "xpost_jump": self.xpost_jump,
            "rho_neg": self.rho_neg,
            "rho_neg2": self.rho_neg2,
            "rho_init": self.rho_init,
            "tau_rho": self.tau_rho,
            "thr_post": self.thr_post,
            "thr_pre": self.thr_pre,
            "thr_b_rho": self.thr_b_rho,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "alpha": self.alpha,
            "beta": self.beta,
            "xpre_factor": self.xpre_factor,
            "w_max": self.w_max,
            "xpre_min": self.xpre_min,
            "xpost_min": self.xpost_min,
            "xpost_max": self.xpost_max,
            "xpre_max": self.xpre_max,
            "tau_xstop": self.tau_xstop,
            "xstop_jump": self.xstop_jump,
            "xstop_max": self.xstop_max,
            "xstop_min": self.xstop_min,
            "thr_stop_h": self.thr_stop_h,
            "thr_stop_l": self.thr_stop_l,
            "U": self.U,
            "tau_d": self.tau_d,
            "tau_f": self.tau_f,
            "k": self.k,
        }

        return self.namespace

    # 2 ------ getters

    # 2.1 ------ network

    def get_sim_data_path(self):
        return self.net_sim_data_path

    # 2.2 ------ neurons

    def get_E_rates(self):
        return [self.E_rec_rate.x[:], self.E_rec_rate.rate[:]]

    """
    Returns a 2D array containing recorded spikes from E population:
    - spike's neuron ids (return[0])
    - spike's times (return[1])
    """

    def get_E_spks(self, spike_trains=False):
        if not spike_trains:
            return [self.E_mon.t[:], self.E_mon.i[:]]
        else:
            if self.dumped_mons_dict:
                from collections import OrderedDict

                spike_trains = {}
                for i, t in zip(
                        self.dumped_mons_dict["E_mon"]["i"],
                        self.dumped_mons_dict["E_mon"]["t"],
                ):
                    if i not in spike_trains:
                        spike_trains[i] = [t]
                    else:
                        spike_trains[i].append(t)

                for i in range(self.N_e):
                    if i not in spike_trains:
                        spike_trains[i] = []

                return OrderedDict(sorted(spike_trains.items()))
            else:
                return self.E_mon.spike_trains()

    """
    Returns a 2D array containing recorded spikes from I population:
    - spike's neuron ids (return[0])
    - spike's times (return[1])
    """

    def get_I_spks(self):
        return [self.I_mon.t[:], self.I_mon.i[:]]

    """
    Returns a 2D array containing recorded spikes from input to the E population:
    - spike's neuron ids (return[0])
    - spike's times (return[1])
    """

    def get_Einp_spks(self):
        return [self.Input_to_E_mon.t[:], self.Input_to_E_mon.i[:]]

    """
    Returns a 2D array containing recorded spikes from input to the I population:
    - spike's neuron ids (return[0])
    - spike's times (return[1])
    """

    def get_Iinp_spks(self):
        return [self.Input_to_I_mon.t[:], self.Input_to_I_mon.i[:]]

    """
    """

    def get_target_spks(self, targets_E=[], targets_I=[], all=False):
        if all == False:
            targeted_E_list = []
            for n_id in targets_E:
                targeted_E_list.append({"id": n_id, "spk_t": []})

            for n_id in range(0, len(self.E_mon.i)):
                if self.E_mon.i[n_id] in targets_E:
                    for y in targeted_E_list:
                        if y["id"] == self.E_mon.i[n_id]:
                            y["spk_t"].append(self.E_mon.t[n_id])

            targeted_I_list = []
            for n_id in targets_I:
                targeted_I_list.append({"id": n_id, "spk_t": []})

            for n_id in range(0, len(self.I_mon.i)):
                if self.I_mon.i[n_id] in targets_I:
                    for y in targeted_I_list:
                        if y["id"] == self.I_mon.i[n_id]:
                            y["spk_t"].append(self.I_mon.t[n_id])

            fn = os.path.join(self.net_sim_data_path, "targeted_E_n_I_spks.pickle")

            with open(fn, "wb") as f:
                pickle.dump((targeted_E_list, targeted_I_list), f)
        else:
            targeted_E_list = []
            for n_id in range(0, self.N_e):
                targeted_E_list.append({"id": n_id, "spk_t": []})

            for n_id in range(0, len(self.E_mon.i)):
                for y in targeted_E_list:
                    if y["id"] == self.E_mon.i[n_id]:
                        y["spk_t"].append(self.E_mon.t[n_id])

            targeted_I_list = []
            for n_id in range(0, self.N_i):
                targeted_I_list.append({"id": n_id, "spk_t": []})

            for n_id in range(0, len(self.I_mon.i)):
                for y in targeted_I_list:
                    if y["id"] == self.I_mon.i[n_id]:
                        y["spk_t"].append(self.I_mon.t[n_id])

            fn = os.path.join(self.net_sim_data_path, "targeted_all_E_n_I_spks.pickle")

            with open(fn, "wb") as f:
                pickle.dump((targeted_E_list, targeted_I_list), f)

    """
    Retrieves the spikes recorded during simulation only from neurons receiving input simuli.
    len(self.E_E.i) -> pre neurons
    len(self.E_E.j) -> post neurons
    """

    def get_spks_from_pattern_neurons(self, export=False):
        spk_mon_ids = []
        spk_mon_ts = []

        if self.dumped_mons_dict:
            data = self.dumped_mons_dict["E_mon"]["i"]
            time = self.dumped_mons_dict["E_mon"]["t"]
        else:
            data = self.E_mon.i[:]
            time = self.E_mon.t[:]

        for x in range(len(data)):
            if data[x] in self.stimulus_neurons_e_ids:
                spk_mon_ids.append(data[x])
                spk_mon_ts.append(time[x] / second)

        if export:
            fn = os.path.join(self.net_sim_data_path, "spks_neurs_with_input.pickle")

            with open(fn, "wb") as f:
                pickle.dump((spk_mon_ids, spk_mon_ts, self.t_run), f)

        return spk_mon_ids, spk_mon_ts

    # 2.3 ------ synapses

    """
    """

    def get_E_E_weights(self):
        return self.E_E_rec.w

    """
    """

    def get_E_E_rho(self):
        return self.E_E_rec.rho[:]

    """
    """

    def get_E_E_xpre(self):
        return self.E_E_rec.xpre[:]

    """
    """

    def get_E_E_xpost(self):
        return self.E_E_rec.xpost[:]

    """
    """

    def get_E_E_x(self):
        return self.E_E_rec.x_[:]

    """
    """

    def pickle_E_E_u_active_inp(self):
        us = []
        count = 0
        avg = 0.0

        for aaa in self.E_E_rec.u:
            if np.mean(aaa) > 0.3 and np.mean(aaa) != avg:
                us.append(aaa)
                avg = np.mean(aaa)

                count += 1
            # if count > 5:
            #     break

        fn = os.path.join(self.net_sim_data_path, "stimulus_neur_u.pickle")

        with open(fn, "wb") as f:
            pickle.dump((us, self.E_E_rec.t / ms), f)

    """
    """

    def pickle_E_E_x_active_inp(self):
        x_ = []
        count = 0
        sum_ = 0.0

        for aaa in self.E_E_rec.x_:
            if np.sum(aaa) / len(aaa) < 1.0 and np.sum(aaa) / len(aaa) != sum_:
                x_.append(aaa)
                sum_ = np.sum(aaa) / len(aaa)

            #     count += 1
            # if count > 5:
            #     break

        fn = os.path.join(self.net_sim_data_path, "stimulus_neur_x.pickle")

        with open(fn, "wb") as f:
            pickle.dump((x_, self.E_E_rec.t / ms), f)

    def get_spikes_pyspike(self, export=False):
        from pyspike import SpikeTrain

        """
        Retrieves the spikes recorded during simulation and saves them in the format expected by PySpike.
        """
        if export:
            fn = os.path.join(self.net_sim_data_path, "spikes_pyspike.txt")

            with open(fn, "w") as file:
                file.write("# Spikes from excitatory population\n\n")
                for v in self.get_E_spks(spike_trains=True).values():
                    np.savetxt(file, np.array(v), newline=" ")
                    file.write("\n")
        else:
            self.pyspike_spks = []

            for v in self.get_E_spks(spike_trains=True).values():
                self.pyspike_spks.append(SpikeTrain(np.array(v), edges=(0, self.net.t)))

    """
    """

    def find_ps(self, attractor, threshold=0.7, verbose=False):
        from scipy.ndimage.filters import uniform_filter1d
        from pyspike import spike_sync_profile
        from helper_functions.other import contiguous_regions

        if attractor[0] in self.pss:
            return (
                self.x_pss[attractor[0]],
                self.y_pss[attractor[0]],
                self.y_smooth_pss[attractor[0]],
                self.pss[attractor[0]],
            )

        if not self.pyspike_spks:
            self.get_spikes_pyspike()

        spike_sync_profile = spike_sync_profile(self.pyspike_spks, indices=attractor[1])

        x, y = spike_sync_profile.get_plottable_data()
        # mean_filter_size = round( len( x ) / 10 )
        mean_filter_size = 20

        try:
            y_smooth = uniform_filter1d(y, size=mean_filter_size)
        except:
            y_smooth = np.zeros(len(x))

        # if there are no spikes we need to force it to count zero PSs
        if np.all([len(i) == 0 for i in self.pyspike_spks]):
            pss = np.array([[]])
        else:
            pss = contiguous_regions(y_smooth > threshold)

        # temporal filter removing PSs that are too close to each other
        try:
            pss = pss[np.insert(np.diff(np.ravel(pss))[1::2] > 50, 0, True), :]
        except:
            pass

        (
            self.x_pss[attractor[0]],
            self.y_pss[attractor[0]],
            self.y_smooth_pss[attractor[0]],
            self.pss[attractor[0]],
        ) = (x, y, y_smooth, pss)

        if verbose:
            for ps in pss:
                print(
                    f"Found PS in {attractor[0]} "
                    f"between {x[ps[0]]} s and {x[ps[1]]} s "
                    f"centered at {x[ps[0] + np.argmax(y_smooth[ps[0]:ps[1]])]} s "
                    f"with max value {np.max(y_smooth[ps[0]:ps[1]])} "
                    f"(mean={np.mean(y_smooth[ps[0]:ps[1]])}, "
                    f"std={np.std(y_smooth[ps[0]:ps[1]])}"
                )

        return x, y, y_smooth, pss

    def get_conn_matrix(self, pop="E_E"):
        if pop == "E_E":
            ee_conn_matrix = np.full((len(self.E), len(self.E)), -1.0)
            ee_conn_matrix[self.E_E.i, self.E_E.j] = 1.0
        else:
            ee_conn_matrix = np.full((len(self.E), len(self.E)), -1.0)
            ee_conn_matrix[self.E_E.i, self.E_E.j] = 1.0

        return ee_conn_matrix

    """
    Pickle to file current state of synaptic matrix.
    """

    def pickle_E_E_syn_matrix_state(self):
        synaptic_matrix = np.full((len(self.E), len(self.E)), -1.0)
        synaptic_matrix[self.E_E.i, self.E_E.j] = self.E_E.rho

        E_E_syn_matrix_path = os.path.join(self.net_sim_data_path, "E_E_syn_matrix")

        if not (os.path.isdir(E_E_syn_matrix_path)):
            os.mkdir(E_E_syn_matrix_path)

        self.E_E_syn_matrix_path = E_E_syn_matrix_path

        fn = os.path.join(self.E_E_syn_matrix_path, "0_E_E_syn_matrix.pickle")

        with open(fn, "wb") as f:
            pickle.dump((synaptic_matrix), f)

    """
    Retrieves the synaptic traces recorded during simulation only from synapses connecting (both) neurons part
    receiving input simuli.
    len(self.E_E.i) -> pre neurons
    len(self.E_E.j) -> post neurons
    self.E_E_rec[self.E_E[0]] -> 1st synapse
    """

    def get_syn_traces_from_pattern_neurons(self):
        syns_neurs_with_input = []

        for x in range(0, len(self.E_E)):
            if (
                    self.E_E.i[x] in self.stimulus_neurons_e_ids
                    and self.E_E.j[x] in self.stimulus_neurons_e_ids
            ):
                temp_dict = {
                    "pre": self.E_E.i[x],
                    "post": self.E_E.j[x],
                    "syn_trace": self.E_E_rec[self.E_E[x]].rho,
                }

                syns_neurs_with_input.append(temp_dict)

        fn = os.path.join(self.net_sim_data_path, "syns_neurs_with_input.pickle")

        with open(fn, "wb") as f:
            pickle.dump(
                (syns_neurs_with_input, self.E_E_rec.t / second, self.thr_b_rho), f
            )

    """
    Retrieves the neurotransmitter traces (x_) recorded during simulation only from synapses connecting (both)
    neurons part receiving input simuli.
    len(self.E_E.i) -> pre neurons
    len(self.E_E.j) -> post neurons
    self.E_E_rec[self.E_E[0]] -> 1st synapse
    """

    def get_x_traces_from_pattern_neurons(self, export=False):
        x_traces = {}

        for x in range(0, len(self.E_E)):
            if (
                    self.E_E.i[x] in self.stimulus_neurons_e_ids
                    and self.E_E.j[x] in self.stimulus_neurons_e_ids
            ):
                x_traces[self.E_E.i[x], self.E_E.j[x]] = self.E_E_rec[self.E_E[x]].x_

        if export:
            fn = os.path.join(self.net_sim_data_path, "xs_neurs_with_input.pickle")

            with open(fn, "wb") as f:
                pickle.dump(
                    (
                        x_traces,
                        self.E_E_rec.t / second,
                        self.tau_d,
                        # self.stimulus_pulse_duration
                    ),
                    f,
                )

        return [x_traces, self.E_E_rec.t / second, self.tau_d]

    """
    Retrieves the utilization traces (u) recorded during simulation only from synapses connecting (both) neurons part
    receiving input simuli.
    len(self.E_E.i) -> pre neurons
    len(self.E_E.j) -> post neurons
    self.E_E_rec[self.E_E[0]] -> 1st synapse
    """

    def get_u_traces_from_pattern_neurons(self, export=False):
        u_traces = {}

        for x in range(0, len(self.E)):
            if x in self.stimulus_neurons_e_ids:
                u_traces[x] = self.E_rec[x].u

        if export:
            fn = os.path.join(self.net_sim_data_path, "us_neurs_with_input.pickle")

            with open(fn, "wb") as f:
                pickle.dump(
                    (
                        u_traces,
                        self.E_E_rec.t / second,
                        self.U,
                        self.tau_f,
                        # self.stimulus_pulse_duration
                    ),
                    f,
                )

        return [u_traces, self.E_E_rec.t / second, self.U, self.tau_d]

    def get_u_traces_from_pattern_neurons_in_range(self, t_window):
        uTracesValues = []

        for x in range(0, len(self.E)):
            if x in self.stimulus_neurons_e_ids:

                for i in range(len(self.E_rec[x].u)):

                    if (
                            self.E_rec.t[i] / second >= t_window[0]
                            and self.E_rec.t[i] / second <= t_window[1]
                    ):
                        uTracesValues.append(self.E_rec[x].u[i])

        fn = os.path.join(self.net_sim_data_path, "attractor_uTraceValues.pickle")

        with open(fn, "wb") as f:
            pickle.dump((uTracesValues), f)

    def get_x_traces_from_pattern_neurons_in_range(self, t_window):
        uTracesValues = []

        for x in range(0, len(self.E)):
            if x in self.stimulus_neurons_e_ids:

                for i in range(len(self.E_rec[x].x_)):

                    if (
                            self.E_rec.t[i] / second >= t_window[0]
                            and self.E_rec.t[i] / second <= t_window[1]
                    ):
                        uTracesValues.append(self.E_rec[x].x_[i])

        fn = os.path.join(self.net_sim_data_path, "attractor_xTraceValues.pickle")

        with open(fn, "wb") as f:
            pickle.dump((uTracesValues), f)
