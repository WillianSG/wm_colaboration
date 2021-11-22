# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- 

Script arguments:
-

Script output:
-

TO DO:
- [X] save connection matrices method.
- [X] snapshots of syn. matrix during simulation method.
- [] load simulation parameters from file (?).
- [X] make 'get_E_E_conn_matrix' general purpose.
- [] method to save network configuration (param. set, pop. params., etc)
"""
import setuptools
from time import localtime, strftime
import os, pickle

from brian2 import *
import numpy as np

# import os, sys, pickle, shutil
# import random
# from time import localtime

prefs.codegen.target = 'numpy'

from load_rule_parameters import *
from load_synapse_model import *
from load_stimulus import *


class RecurrentCompetitiveNet:
    def __init__(self, plasticity_rule='LR2', parameter_set='2.4', t_run=2 * second):

        # ------ simulation parameters
        self.net_id = strftime("%d%b%Y_%H-%M-%S", localtime())

        self.net_sim_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '../')))

        self.net_sim_data_path = ''

        self.t_run = t_run
        self.dt = 0.1 * ms

        self.int_meth_neur = 'linear'
        self.int_meth_syn = 'euler'

        # ------ network parameters
        self.stimulus_pulse = False
        self.stimulus_pulse_duration = self.t_run - 1 * second
        self.stimulus_pulse_clock_dt = 0.1 * ms

        self.stim_size_e = 64
        self.stim_size_i = 64

        # ------ neurons parameters
        self.neuron_type = 'LIF'

        self.N_input_e = 256  # num. of input neurons for E
        self.N_input_i = 64  # num. of input neurons for I

        # excitatory population

        self.stim_freq_e = 6600 * Hz
        self.stim_freq_i = 3900 * Hz

        self.N_e = 256  # num. of neurons
        self.Vr_e = -65 * mV  # resting potential
        self.Vrst_e = -65 * mV  # reset potential
        self.Vth_e_init = -52 * mV  # initial threshold voltage
        self.Vth_e_incr = 5 * mV  # post-spike threshold voltage increase
        self.tau_Vth_e = 20 * ms  # time constant of threshold decay
        self.taum_e = 20 * ms  # membrane time constant
        self.tref_e = 2 * ms  # refractory period
        self.tau_epsp_e = 3.5 * ms  # time constant of EPSP
        self.tau_ipsp_e = 5.5 * ms  # time constant of IPSP

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
        self.syn_delay_Vepsp_e_e = 0 * ms
        self.syn_delay_w_update_e_e = 0 * ms

        # weights
        self.w_e_e_max = 7.5 * mV
        self.w_i_i_max = 7.5 * mV

        self.w_input_e = 1 * mV
        self.w_input_i = 1 * mV
        self.w_e_i = 1 * mV
        self.w_i_e = 1 * mV
        self.w_e_e = 0.5 * mV
        self.w_i_i = 0.5 * mV

        # data save
        self.M_ee = []

        self.E_E_syn_matrix_snaptshot = True
        self.E_E_syn_matrix_snaptshot_dt = 100 * ms
        self.E_E_syn_matrix_path = ''

        # ------ data (monitors) parameters
        self.rec_dt = 0.1 * ms

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

        self.Input_E_rec_attributes = ('w')
        self.Input_I_rec_attributes = ('w')
        self.E_rec_attributes = ('Vm', 'Vepsp', 'Vipsp')
        self.I_rec_attributes = ('Vm', 'Vepsp', 'Vipsp')
        self.E_E_rec_attributes = ['w']
        self.E_I_rec_attributes = ('w')
        self.I_E_rec_attributes = ('w')

        if self.plasticity_rule == 'LR4':
            self.E_E_rec_attributes.append('x_')
            self.E_E_rec_attributes.append('u')

        # ------ misc operation variables
        self.stimulus_neurons_e_ids = []
        self.stimulus_neurons_i_ids = []

    # 1 ------ setters

    # 1.1 ------ neurons

    """
    Sets the objects representing the excitatory (E), inhibitory (I) and input neuron populations used in the model.
    - LIF model with adaptive threshold.
    """

    def set_neuron_pop(self):
        # equations (voltage based neuron model)
        self.eqs_e = Equations('''
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / taum_e : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt
			dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt''',
                               Vr_e=self.Vr_e,
                               taum_e=self.taum_e,
                               tau_epsp=self.tau_epsp_e,
                               tau_ipsp=self.tau_ipsp_e,
                               Vth_e_init=self.Vth_e_init,
                               tau_Vth_e=self.tau_Vth_e)

        self.eqs_i = Equations(''' 
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_i)) / taum_i : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt''',
                               Vr_i=self.Vr_i,
                               taum_i=self.taum_i,
                               tau_epsp=self.tau_epsp_i,
                               tau_ipsp=self.tau_ipsp_i)

        # populations
        self.E = NeuronGroup(N=self.N_e, model=self.eqs_e,
                             reset='''Vm = Vrst_e 
					Vth_e += Vth_e_incr''',
                             threshold='Vm > Vth_e',
                             refractory=self.tref_e,
                             method=self.int_meth_neur,
                             name='E')

        self.I = NeuronGroup(N=self.N_i, model=self.eqs_i,
                             reset='Vm = Vrst_i',
                             threshold='Vm > Vth_i',
                             refractory=self.tref_i,
                             method=self.int_meth_neur,
                             name='I')

        self.Input_to_E = NeuronGroup(N=self.N_input_e,
                                      model='rates : Hz',
                                      threshold='rand()<rates*dt',
                                      name='Input_to_E')

        self.Input_to_I = NeuronGroup(N=self.N_input_i,
                                      model='rates : Hz',
                                      threshold='rand()<rates*dt',
                                      name='Input_to_I')

        # populations's attributes
        self.E.Vth_e = self.Vth_e_init

        # rand init membrane voltages
        self.E.Vm = (self.Vrst_e + rand(self.N_e) * (self.Vth_e_init - self.Vr_e))
        self.I.Vm = (self.Vrst_i + rand(self.N_i) * (self.Vth_i - self.Vr_i))

    # 1.2 ------ synapses

    """
    Loads synaptic rule equations.
    """

    def set_learning_rule(self):
        # rule's equations
        [self.model_E_E,
         self.pre_E_E,
         self.post_E_E] = load_synapse_model(
            plasticity_rule=self.plasticity_rule,
            neuron_type=self.neuron_type,
            bistability=self.bistability,
            stoplearning=self.stop_learning,
            resources=self.resources)

        # rule's parameters
        [self.tau_xpre,
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
         self.tau_f] = load_rule_parameters(
            self.plasticity_rule,
            self.rule_parameters)

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

        self.I_I = Synapses(  # I-I plastic synapses
            source=self.I,
            target=self.I,
            model=self.model_E_E,
            on_pre=self.pre_E_E,
            on_post=self.post_E_E,
            method=self.int_meth_syn,
            name='I_I')

        # connecting synapses
        self.Input_E.connect(j='i')
        self.Input_I.connect(j='i')

        self.E_I.connect(True, p=self.p_e_i)
        self.I_E.connect(True, p=self.p_i_e)
        self.E_E.connect('i!=j', p=self.p_e_e)
        self.I_I.connect('i!=j', p=self.p_i_i)

        # init synaptic variables
        self.Input_E.w = self.w_input_e
        self.Input_I.w = self.w_input_i
        self.E_I.w = self.w_e_i
        self.I_E.w = self.w_i_e
        self.E_E.w = self.w_e_e
        self.I_I.w = self.w_i_i

        if self.plasticity_rule == 'LR4':
            self.E_E.x_ = 1.0
            self.E_E.u = self.U

        self.E_E.Vepsp_transmission.delay = self.syn_delay_Vepsp_e_e

    """
    Allows state variables change in synaptic model.
    """

    def set_E_E_plastic(self, plastic=False):
        self.E_E.plastic = plastic

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
                if isnan(self.M_ee[pre_id][post_id]) == False:
                    percent_ = uniform(0, 1)
                    if percent_ < percent:
                        s = uniform(0, 1)
                        self.E_E.rho[pre_id, post_id] = round(s, 2)
                    else:
                        self.E_E.rho[pre_id, post_id] = 0.0

    # 1.4 ------ network operation

    """
    """

    def run_net(self, report='stdout', period=1):
        self.net.run(
            self.t_run,
            report=report,
            report_period=period * second,
            namespace=self.set_net_namespace())

        self.net.stop()

    """
    """

    def set_stimulus_e(self, stimulus, frequency, offset=0, stimulus_size=0):
        if stimulus_size == 0:
            stimulus_size = self.stim_size_e

        if stimulus != '':
            self.stimulus_neurons_e_ids = load_stimulus(
                stimulus_type=stimulus,
                stimulus_size=stimulus_size,
                offset=offset)

            self.Input_to_E.rates[self.stimulus_neurons_e_ids] = frequency
        else:
            self.Input_to_E.rates[self.stimulus_neurons_e_ids] = frequency

    """
    """

    def set_stimulus_i(self, stimulus, frequency, offset=0):
        if stimulus != '':
            self.stimulus_neurons_i_ids = load_stimulus(
                stimulus_type=stimulus,
                stimulus_size=self.stim_size_i,
                offset=offset)

            self.Input_to_I.rates[self.stimulus_neurons_i_ids] = frequency
        else:
            self.Input_to_I.rates[self.stimulus_neurons_i_ids] = frequency

    # 1.3 ------ network initializers

    """
    Sets and configures the rcn network objects for simulation.
    """

    def net_init(self):
        self.set_results_folder()  # sim. results
        self.set_learning_rule()  # rule eqs./params.
        self.set_neuron_pop()  # neuron populations
        self.set_synapses()  # syn. connections
        self.set_spk_monitors()
        self.set_state_monitors()

        self.set_net_obj()

    """
    Creates a folder for simulation results in the root directory.
    """

    def set_results_folder(self):
        network_sim_dir = os.path.join(
            self.net_sim_path,
            'network_results')

        if not (os.path.isdir(network_sim_dir)):
            os.mkdir(network_sim_dir)

        network_sim_dir = os.path.join(
            network_sim_dir,
            self.net_id + '_RCN')

        if not (os.path.isdir(network_sim_dir)):
            os.mkdir(network_sim_dir)

        self.net_sim_data_path = network_sim_dir

        if self.E_E_syn_matrix_snaptshot:
            E_E_syn_matrix_path = os.path.join(
                self.net_sim_data_path,
                'E_E_syn_matrix')

            if not (os.path.isdir(E_E_syn_matrix_path)):
                os.mkdir(E_E_syn_matrix_path)

            self.E_E_syn_matrix_path = E_E_syn_matrix_path

    """
    """

    def set_spk_monitors(self):
        self.Input_to_E_mon = SpikeMonitor(
            source=self.Input_to_E,
            record=self.Input_to_E_mon_record,
            name='Input_to_E_mon')

        self.Input_to_I_mon = SpikeMonitor(
            source=self.Input_to_I,
            record=self.Input_to_I_mon_record,
            name='Input_to_I_mon')

        self.E_mon = SpikeMonitor(
            source=self.E,
            record=self.E_mon_record,
            name='E_mon')

        self.I_mon = SpikeMonitor(
            source=self.I,
            record=self.I_mon_record,
            name='I_mon')

    """
    """

    def set_state_monitors(self):
        self.Input_E_rec = StateMonitor(source=self.Input_E,
                                        variables=self.Input_E_rec_attributes,
                                        record=self.Input_E_rec_record,
                                        dt=self.rec_dt,
                                        name='Input_E_rec')

        self.Input_I_rec = StateMonitor(source=self.Input_I,
                                        variables=self.Input_I_rec_attributes,
                                        record=self.Input_I_rec_record,
                                        dt=self.rec_dt,
                                        name='Input_I_rec')

        self.E_rec = StateMonitor(source=self.E,
                                  variables=self.E_rec_attributes,
                                  record=self.E_rec_record,
                                  dt=self.rec_dt,
                                  name='E_rec')

        self.I_rec = StateMonitor(source=self.I,
                                  variables=self.I_rec_attributes,
                                  record=self.I_rec_record,
                                  dt=self.rec_dt,
                                  name='I_rec')

        self.E_E_rec = StateMonitor(source=self.E_E,
                                    variables=self.E_E_rec_attributes,
                                    record=self.E_E_rec_record,
                                    dt=self.rec_dt,
                                    name='E_E_rec')

        self.E_I_rec = StateMonitor(source=self.E_I,
                                    variables=self.E_I_rec_attributes,
                                    record=self.E_I_rec_record,
                                    dt=self.rec_dt,
                                    name='E_I_rec')

        self.I_E_rec = StateMonitor(source=self.I_E,
                                    variables=self.I_E_rec_attributes,
                                    record=self.I_E_rec_record,
                                    dt=self.rec_dt,
                                    name='I_E_rec')

    """
    Creates a brian2 network object with the neuron/synapse objects defined.
    """

    def set_net_obj(self):
        self.stimulus_pulse_clock = Clock(
            self.stimulus_pulse_clock_dt,
            name='stim_pulse_clk')

        self.E_E_syn_matrix_clock = Clock(
            self.E_E_syn_matrix_snaptshot_dt,
            name='E_E_syn_matrix_clk')

        if self.stimulus_pulse:
            @network_operation(clock=self.stimulus_pulse_clock)
            def stimulus_pulse():
                if defaultclock.t >= self.stimulus_pulse_duration:
                    self.set_stimulus_e(stimulus='', frequency=0 * Hz)
        else:
            @network_operation(clock=self.stimulus_pulse_clock)
            def stimulus_pulse():
                pass

        if self.E_E_syn_matrix_snaptshot:
            @network_operation(clock=self.E_E_syn_matrix_clock)
            def store_E_E_syn_matrix_snapshot():
                synaptic_matrix = np.full((len(self.E), len(self.E)), -1.0)
                synaptic_matrix[self.E_E.i, self.E_E.j] = self.E_E.rho

                fn = os.path.join(
                    self.E_E_syn_matrix_path,
                    str(int(round(defaultclock.t / ms, 0))) + '_ms_E_E_syn_matrix.pickle')

                with open(fn, 'wb') as f:
                    pickle.dump((
                        synaptic_matrix), f)
        else:
            @network_operation(clock=self.E_E_syn_matrix_clock)
            def store_E_E_syn_matrix_snapshot():
                pass

        defaultclock.dt = self.dt

        self.net = Network(
            self.Input_to_E,
            self.Input_to_I,
            self.E,
            self.I,
            self.Input_E,
            self.Input_I,
            self.E_I,
            self.I_E,
            self.E_E,
            self.I_I,
            self.Input_to_E_mon,
            self.Input_to_I_mon,
            self.E_mon,
            self.I_mon,
            self.Input_E_rec,
            self.Input_I_rec,
            self.E_rec,
            self.I_rec,
            self.E_E_rec,
            self.E_I_rec,
            self.I_E_rec,
            stimulus_pulse,
            store_E_E_syn_matrix_snapshot,
            name='rcn_net')

    """
    """

    def set_net_namespace(self):
        self.namespace = {
            'Vrst_e': self.Vrst_e,
            'Vth_e_init': self.Vth_e_init,
            'Vrst_i': self.Vrst_i,
            'Vth_i': self.Vth_i,
            'Vth_e_incr': self.Vth_e_incr,
            'tau_xpre': self.tau_xpre,
            'tau_xpost': self.tau_xpost,
            'xpre_jump': self.xpre_jump,
            'xpost_jump': self.xpost_jump,
            'rho_neg': self.rho_neg,
            'rho_neg2': self.rho_neg2,
            'rho_init': self.rho_init,
            'tau_rho': self.tau_rho,
            'thr_post': self.thr_post,
            'thr_pre': self.thr_pre,
            'thr_b_rho': self.thr_b_rho,
            'rho_min': self.rho_min,
            'rho_max': self.rho_max,
            'alpha': self.alpha,
            'beta': self.beta,
            'xpre_factor': self.xpre_factor,
            'w_max': self.w_max,
            'xpre_min': self.xpre_min,
            'xpost_min': self.xpost_min,
            'xpost_max': self.xpost_max,
            'xpre_max': self.xpre_max,
            'tau_xstop': self.tau_xstop,
            'xstop_jump': self.xstop_jump,
            'xstop_max': self.xstop_max,
            'xstop_min': self.xstop_min,
            'thr_stop_h': self.thr_stop_h,
            'thr_stop_l': self.thr_stop_l,
            'U': self.U,
            'tau_d': self.tau_d,
            'tau_f': self.tau_f}

        return self.namespace

    # 2 ------ getters

    # 2.1 ------ network

    def get_sim_data_path(self):
        return self.net_sim_data_path

    # 2.2 ------ neurons

    """
    Returns a 2D array containing recorded spikes from E population:
    - spike's neuron ids (return[0])
    - spike's times (return[1])
    """

    def get_E_spks(self):
        return [self.E_mon.t[:], self.E_mon.i[:]]

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
                targeted_E_list.append({'id': n_id, 'spk_t': []})

            for n_id in range(0, len(self.E_mon.i)):
                if (self.E_mon.i[n_id] in targets_E):
                    for y in targeted_E_list:
                        if y['id'] == self.E_mon.i[n_id]:
                            y['spk_t'].append(self.E_mon.t[n_id])

            targeted_I_list = []
            for n_id in targets_I:
                targeted_I_list.append({'id': n_id, 'spk_t': []})

            for n_id in range(0, len(self.I_mon.i)):
                if (self.I_mon.i[n_id] in targets_I):
                    for y in targeted_I_list:
                        if y['id'] == self.I_mon.i[n_id]:
                            y['spk_t'].append(self.I_mon.t[n_id])

            fn = os.path.join(
                self.net_sim_data_path,
                'targeted_E_n_I_spks.pickle')

            with open(fn, 'wb') as f:
                pickle.dump((
                    targeted_E_list,
                    targeted_I_list), f)
        else:
            targeted_E_list = []
            for n_id in range(0, self.N_e):
                targeted_E_list.append({'id': n_id, 'spk_t': []})

            for n_id in range(0, len(self.E_mon.i)):
                for y in targeted_E_list:
                    if y['id'] == self.E_mon.i[n_id]:
                        y['spk_t'].append(self.E_mon.t[n_id])

            targeted_I_list = []
            for n_id in range(0, self.N_i):
                targeted_I_list.append({'id': n_id, 'spk_t': []})

            for n_id in range(0, len(self.I_mon.i)):
                for y in targeted_I_list:
                    if y['id'] == self.I_mon.i[n_id]:
                        y['spk_t'].append(self.I_mon.t[n_id])

            fn = os.path.join(
                self.net_sim_data_path,
                'targeted_all_E_n_I_spks.pickle')

            with open(fn, 'wb') as f:
                pickle.dump((
                    targeted_E_list,
                    targeted_I_list), f)

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
        sum_ = 0.0

        for aaa in self.E_E_rec.u:
            if np.sum(aaa) / len(aaa) > 0.3 and np.sum(aaa) / len(aaa) != sum_:
                us.append(aaa)
                sum_ = np.sum(aaa) / len(aaa)

                count += 1
            if count > 10:
                break

        fn = os.path.join(self.net_sim_data_path, 'stimulus_neur_u.pickle')

        with open(fn, 'wb') as f:
            pickle.dump((
                us,
                self.E_E_rec.t / ms), f)

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

                count += 1
            if count > 10:
                break

        fn = os.path.join(self.net_sim_data_path, 'stimulus_neur_x.pickle')

        with open(fn, 'wb') as f:
            pickle.dump((
                x_,
                self.E_E_rec.t / ms), f)

    """
    """

    def get_conn_matrix(self, pop='E_E'):
        if pop == 'E_E':
            ee_conn_matrix = np.full((len(self.E), len(self.E)), -1.0)
            ee_conn_matrix[self.E_E.i, self.E_E.j] = 1.0
        else:
            ee_conn_matrix = np.full((len(self.E), len(self.E)), -1.0)
            ee_conn_matrix[self.E_E.i, self.E_E.j] = 1.0

        return ee_conn_matrix
