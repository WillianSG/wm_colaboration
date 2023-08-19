# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""
import sys, os, pickle
from brian2 import Hz, mV, second, ms, prefs
from datetime import datetime

prefs.codegen.target = 'numpy'

if sys.platform in ['linux', 'win32']:
    sys.path.append(os.path.abspath(os.path.join(__file__ , '../../')))
    from recurrent_competitive_network import RecurrentCompetitiveNet
    from firing_rate_histograms import firing_rate_histograms
else:
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
    from helper_functions.firing_rate_histograms import firing_rate_histograms

class sFSA:
    def __init__(self, FSA_model: dict, args):

        self.sFSA_id = str(datetime.utcnow()).replace(' ', '_').replace(':', '-').replace('.', '')

        self.plasticity_rule = 'LR4'
        self.parameter_set = '2.2'
        self.attractor_size = 64

        self.__FSA_model = FSA_model
        self.__args = args

        self.__RCN = RecurrentCompetitiveNet(self.plasticity_rule, self.parameter_set)

        self.record_traces = True

        self.start_twindow = (0.2, 0.1)     # 1st: cuing period; 2nd: free net. evolution period.
        self.input_twindow = (0.8, 4.2)
        self.free_activity = 0.7            # free net. evolution after last input toke.     

        self.__loadSfsaArgs()
        self.__loadSfsaModel()
        self.__enterStartState()

        self.inputed_sequence = []
        self.data_folder = ''

    # - Private Functions -

    def __loadSfsaArgs(self):
            '''
            Initializes some network hyperparameters relating to attractor and state transition dynamics.
            '''

            self.__RCN.set_learning_rule()  # call it here before net.init() to prevent overriding parameters bellow.

            # - Simulation config. parameters.

            self.__RCN.E_E_syn_matrix_snapshot = False

            # - Attractor dynamics parameters.

            self.__RCN.spont_rate              = self.__args.ba_rate * Hz
            self.__RCN.w_max                   = self.__args.e_e_max_weight * mV
            self.__RCN.w_e_i                   = self.__args.W_ei * mV
            self.__RCN.w_i_e                   = self.__args.W_ie * mV
            self.__RCN.stim_freq_i             = self.__args.inh_rate * Hz
            self.__RCN.tau_Vth_e               = 0.1 * second
            self.__RCN.k                       = 4    
            self.__RCN.U                       = 0.2

            # - State transition dynamics parameters.

            self.__RCN.thr_GO_state            = self.__args.thr_GO_state
            self.__RCN.delay_A2GO              = self.__args.delay_A2GO * second
            self.__RCN.delay_A2B               = self.__args.delay_A2B * second

    def __loadSfsaModel(self):
        '''
            Creates a dictionary that will be used later to initialize the RCN. The dictionary contains the total number of E/I neurons
        necessary to encode each state and input symbol as attractors (the list of neuron ids in each attractor is also created here). The 
        list of state transition rules is parsed to initialize the necessary connections between attractors.
        '''
        
        self.__sfsa_state = {'S': {}, 'I': {}, 'T': {}, 'N_e': 0, 'N_i': 0}

        # - Computing required RCN size (# of attractors) to implement SFA model.

        self.__sfsa_state['N_e'] = int((len(self.__FSA_model['S']) + len(self.__FSA_model['I']))*self.attractor_size)
        self.__sfsa_state['N_i'] = int((self.__sfsa_state['N_e']*25)/100)

        # - Defining attractors' as neuron IDs.

        ids_start = 0
        ids_end = self.attractor_size

        for s in self.__FSA_model['S']:

            self.__sfsa_state['S'][s] = list(range(ids_start, ids_end))

            ids_start += self.attractor_size
            ids_end += self.attractor_size

        for i in self.__FSA_model['I']:

            self.__sfsa_state['I'][i] = list(range(ids_start, ids_end))

            ids_start += self.attractor_size
            ids_end += self.attractor_size

        ids_start = 0

        for t in self.__FSA_model['T']:

            self.__sfsa_state['T'][ids_start] = t

            ids_start += 1

        # - Initialize network based on FSA model.

        self.__initRcn4Fsa()

        # - Configure connectivity for state transtions based on FSA model.

        self.__setStateTransitions()

    def __initRcn4Fsa(self):
        '''
            Configure and initializes the network (RCN) objects necessary to implement the FSA passed as input. This method
        goes over the list of states S and inputs I and creates an attractor (set weights of synapses between E neurons to 
        potentiated) for each of them.
        '''

        if self.record_traces:

            self.__RCN.E_rec_record = True

        # - Setting total pool sizes.
        self.__RCN.N_input_e = self.__sfsa_state['N_e']
        self.__RCN.N_input_i = self.__sfsa_state['N_i']

        self.__RCN.stim_size_e = self.__sfsa_state['N_i']
        self.__RCN.stim_size_i = self.__sfsa_state['N_i']

        self.__RCN.N_e = self.__sfsa_state['N_e']
        self.__RCN.N_i = self.__sfsa_state['N_i']

        self.__RCN.net_init()

        # - Setting inhibitory input and plasticity of connections.

        self.__RCN.set_stimulus_i(stimulus = 'flat_to_I', frequency = self.__args.inh_rate * Hz)
        self.__RCN.set_E_E_plastic(plastic = False)
        self.__RCN.set_E_E_ux_vars_plastic(plastic = True)

        # - Forming attractors (set connection matrix).

        for parameter, values in self.__sfsa_state.items():

            if parameter in ['S', 'I']:

                for token, neur_IDs in values.items():

                    self.__RCN.set_potentiated_synapses(neur_IDs)

    def __setStateTransitions(self):
        '''
            The state transition rules are used to initialize (set potentiated) the synapases between attractors 
        that implement the transition of activity between attractors.
        '''

        for tid, transition in self.__sfsa_state['T'].items():

            # - Parsing transition.
            
            _s0 = transition.split('->')[0].split(', ')[0].replace('(', '')     # current state.
            _i = transition.split('->')[0].split(', ')[1].replace(')', '')      # input token.
            _s1 = transition.split('->')[1]                                     # next state.

            # - Setting connections for state->input->state transitions (transfering cue).
            
            self.__RCN.set_synapses_A_2_B(A_ids = self.__sfsa_state['I'][_i], B_ids = self.__sfsa_state['S'][_s0], weight = self.__args.w_trans*mV)

            # - Setting connections for state-input matching leading to next state.

            if self.__sfsa_state['S'][_s0] != self.__sfsa_state['S'][_s1]:

                # if _i != '0' and _s1 != 'B':
                self.__RCN.set_synapses_A_2_GO(A_ids = self.__sfsa_state['I'][_i], GO_ids = self.__sfsa_state['S'][_s1], weight = self.__args.w_acpt*mV)
                
                # if _s0 != 'A' and _s1 != 'B':
                self.__RCN.set_synapses_A_2_GO(A_ids = self.__sfsa_state['S'][_s0], GO_ids = self.__sfsa_state['S'][_s1], weight = self.__args.w_acpt*mV)

    def __enterStartState(self):
        '''
        Cues the 1st state in the state list to set it active (the first state is assumed to be the FSA's starting state).
        '''

        # - Setting/sending input to attractor representing starting state of the FSA.

        start_state = self.__FSA_model['S'][0]

        act_ids = self.__RCN.generic_stimulus(
                frequency = self.__RCN.stim_freq_e, 
                stim_perc = self.__args.gs_percent, 
                subset = self.__sfsa_state['S'][start_state])

        self.__RCN.run_net(duration = self.start_twindow[0])

        self.__RCN.generic_stimulus_off(act_ids)

        # - Running network without external input (attractor evolution after stimulation).

        self.__RCN.run_net(duration = self.start_twindow[1])

        # @TODO return sequence of activations from network activity.

    def __feedInputToken(self, input_token):
        '''
            Cues an attractor for 0.6s and run the RCN foward in time for 3.4s. A transition will
        (or not) happen after processing the input token, so we simulate a total of 4s from cueing
        to let the activity in the network evolve.
        '''

        print(f'> input: {input_token}')

        self.inputed_sequence.append(input_token)

        # - Setting/sending input to attractor.

        act_ids = self.__RCN.generic_stimulus(
                frequency = self.__RCN.stim_freq_e, 
                stim_perc = self.__args.gs_percent, 
                subset = self.__sfsa_state['I'][input_token])
        
        self.__RCN.run_net(duration = self.input_twindow[0])

        self.__RCN.generic_stimulus_off(act_ids)

        # - Running network without external input (attractor evolution after stimulation).

        self.__RCN.run_net(duration = self.input_twindow[1])

        # @TODO return sequence of activations from network activity.

    def __getMostActiveAttractor(self, spikes_per_state, t_window, attractor_size):
        '''
            Computes the mean rate of each attractor representing a state in S within a simulation time window. The highest 
        mean rate is taken to be currently active attractor (the state in which the FSA is at) and the state S it represents 
        is returned. If the highest mean rate if bellow 5Hz then all attractors are taken to be at spontaneous activity level, 
        in which case 'null' is returned.
        '''
        import numpy as np

        # @TODO: compute threshold to prevent false positives.
        rate_thr = 7                                                                        # threshold for spontaneous activity.

        states_mean_rate = {}

        activations = []

        for state, spikes in spikes_per_state.items():

            spks_in_window = [i for i in spikes if i >= t_window[0] and i <= t_window[1]]

            [_,
            __,
            ___,
            t_hist_fr] = firing_rate_histograms(
                tpoints = np.array(spks_in_window)*second,
                inds = np.arange(0, attractor_size),                                        # not important the correct IDs.
                bin_width = 25*ms,
                N_pop = attractor_size,
                flag_hist = 'time_resolved' )
        
            states_mean_rate[state] = np.round(np.mean(t_hist_fr), 1)

            if states_mean_rate[state] > rate_thr:

                activations.append(states_mean_rate[state])

        # @TODO: check if transition is caused by input alone.

        if len(activations) > 1:                                                            # check if more than one state is active.
            return 'two'
        else:
            if max(zip(states_mean_rate.values(), states_mean_rate.keys()))[0] > rate_thr:  # check if winning state is not at spontaneous activity level.
                return max(zip(states_mean_rate.values(), states_mean_rate.keys()))[1]
            else:
                return 'null'

    # - Public Functions -

    def feedInputWord(self, input_word: list):
        '''
        For each input token in input_word the respective attractor in I is cued.
        '''

        self.inputed_sequence = []

        for token in input_word:

            self.__feedInputToken(token)

        self.__RCN.run_net(duration = self.free_activity)

    def exportSfsaData(self, network_plot = True, sub_dir = '', name_ext = ''):
        '''
            Exports a dictionary containing data from the simulated network to a directory. The network's state history is 
        extracted from the spike trains during simulation and the networks's activity (E spikes raster plot) is also plotted to a file.
        '''

        # - Creating data folder.

        if self.data_folder == '':

            if sub_dir != '':
                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', sub_dir))

                if not (os.path.isdir(data_folder)):
                    os.mkdir(data_folder)

                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', sub_dir, f'sFSA_{self.sFSA_id}'))
            else:
                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', f'sFSA_{self.sFSA_id}'))

            if not (os.path.isdir(data_folder)):
                os.mkdir(data_folder)

            self.data_folder = data_folder

        # - Export metadata.

        with open(os.path.join(self.data_folder, 'parameters.txt'), "w") as file:
            file.write("\nRCN\n")
            for arg in vars(self.__args):
                arg_value = getattr(self.__args, arg)
                if arg_value is not None:
                    file.write(f"\n{arg}: {arg_value}")

            file.write(f"\np_A2GO: {self.__RCN.p_A2GO}")
            file.write(f"\ndelay_A2GO: {self.__RCN.delay_A2GO}")
            file.write(f"\np_A2B: {self.__RCN.p_A2B}")
            file.write(f"\ndelay_A2B: {self.__RCN.delay_A2B}")
            file.write(f"\nstart_twindow: {self.start_twindow}s")
            file.write(f"\ninput_twindow: {self.input_twindow}s")
            file.write(f"\nfree_activity: {self.free_activity}s")
            file.write("\n\nExc. neurons\n")
            file.write(f"\nVr_e: {self.__RCN.Vr_e}")
            file.write(f"\nVrst_e: {self.__RCN.Vrst_e}")
            file.write(f"\nVth_e_init: {self.__RCN.Vth_e_init}")
            file.write(f"\nVth_e_decr: {self.__RCN.Vth_e_decr}")
            file.write(f"\ntau_Vth_e: {self.__RCN.tau_Vth_e}")
            file.write(f"\ntaum_e: {self.__RCN.taum_e}")
            file.write(f"\ntref_e: {self.__RCN.tref_e}")
            file.write(f"\ntau_epsp_e: {self.__RCN.tau_epsp_e}")
            file.write(f"\ntau_ipsp_e: {self.__RCN.tau_ipsp_e}")
            file.write("\n\nInh. neurons\n")
            file.write(f"\nVr_i: {self.__RCN.Vr_i}")
            file.write(f"\nVrst_i: {self.__RCN.Vrst_i}")
            file.write(f"\nVth_i: {self.__RCN.Vth_i}")
            file.write(f"\ntaum_i: {self.__RCN.taum_i}")
            file.write(f"\ntref_i: {self.__RCN.tref_i}")
            file.write(f"\ntau_epsp_i: {self.__RCN.tau_epsp_i}")
            file.write(f"\ntau_ipsp_i: {self.__RCN.tau_ipsp_i}")

        '''
        @TODO: added other network hyperparameters not included in the arguments.
        - neuron equations
        - neurons parameters
        '''

        # - Saving spike trains and traces.

        _E_spk_trains = {key: value for key, value in self.__RCN.E_mon.spike_trains().items()}          # excitatory spike trains.

        sim_t = self.__RCN.E_rec.t/second
        
        states_Vth_traces = {}
        
        if self.record_traces:
            # exporting threshold voltage traces.

            for state, neur_ids in self.__sfsa_state['S'].items():

                states_Vth_traces[state] = self.__RCN.E_rec.Vth_e[neur_ids, :]/mV

        self.sFSA_sim_data = {
            'E_spk_trains': _E_spk_trains,
            'Vth_traces': states_Vth_traces,
            'sim_t': sim_t,
            'sFSA': self.__sfsa_state,
            'input_sequence': self.inputed_sequence,
            'thr_GO_state': self.__args.thr_GO_state,
            'start_twindow': self.start_twindow,
            'input_twindow': self.input_twindow,
            'free_activity': self.free_activity
        }

        # - Compute state history.

        state_history = self.computeStateHistory()
        self.sFSA_sim_data['state_history'] = state_history

        # - Plot network activity.

        if network_plot:

            self.plotSfsaNetwork(name_ext)

        # - Export data.

        fn = os.path.join(
            self.data_folder,
            f'sSFA_simulation_data{name_ext}.pickle')

        with open(fn, 'wb') as f:
            pickle.dump(self.sFSA_sim_data, f)
    
    def computeStateHistory(self, data_path = ''):
        '''
            Parses the networks E spikes so as to divide them into the respective attractors representing the FSA states. The S attractors mean
        rates are computed winthin a time window before each input is fed in order to compute in what state the sFSA was before the input was fed (
        the last active attractor is also computed to get the full history of state transitions).
        '''

        if data_path != '':

            data_file = open(os.path.join(data_path, 'sSFA_simulation_data.pickle'), 'rb')
            simulation_data = pickle.load(data_file)

        else:

            simulation_data = self.sFSA_sim_data

        # - Getting spike trains of neurons per state attractor.

        spikes_per_state = {}

        for S, S_neur_ids in simulation_data['sFSA']['S'].items():                      # for each neuron in an attractor representing a state.

            attractor_size = len(S_neur_ids)                                            # assumes all attractors have the same size.

            spikes_per_state[S] = []                                                    # all state's S spikes.

            for neur_id in simulation_data['E_spk_trains'].keys():                      # for each neuron spike train.

                if neur_id in S_neur_ids:                                               # save spikes if neuron belongs to attractor S.

                    spikes_per_state[S] += list(simulation_data['E_spk_trains'][neur_id]/second)

        # - Getting state transisitons based on mean rate and input tokens.

        state_sequence = []

        S_winning = self.__getMostActiveAttractor(spikes_per_state, (0, 0.1), attractor_size) # starting state.
        state_sequence.append(S_winning)

        _input_twindow = simulation_data['input_twindow'][0] + simulation_data['input_twindow'][1]

        for i in range(len(simulation_data['input_sequence'])):

            if i == len(simulation_data['input_sequence'])-1:

                t_start = simulation_data['sim_t'][-1]-0.7                              # look back .7s before an input token time window.
                t_end = simulation_data['sim_t'][-1]

            else:

                t_end = _input_twindow*(i+1)+(simulation_data['start_twindow'][0] + simulation_data['start_twindow'][1])
                t_start = t_end-0.7

            S_winning = self.__getMostActiveAttractor(spikes_per_state, (t_start, t_end), attractor_size)

            state_sequence.append(S_winning)

        return state_sequence
    
    def plotSfsaNetwork(self, name_ext = ''):
        '''
            Plots (raster plot) the spikes from the E neurons representing input tokens (in black) and states (colored) as well as the voltage
        traces of each attractor in S.
        '''
        import matplotlib.pyplot as plt
        import numpy as np

        # - Create a figure.

        fig, (ax1, ax2) = plt.subplots(nrows = 2, sharex = True, figsize = (15, 5))

        states_colors = ['mediumslateblue', 'magenta', 'darkorange', 'purple', 'b', 'r', 'g', 'darkred']

        # - Axis for the spike raster plot.

        E_spk_trains = self.sFSA_sim_data['E_spk_trains']

        _aux_c = 0

        for _s, neur_IDs in self.sFSA_sim_data['sFSA']['S'].items():

            for neuron_id in neur_IDs:
                if neuron_id in E_spk_trains.keys():
                    spikes = E_spk_trains[neuron_id]
                    y = np.ones_like(spikes) * neuron_id
                    ax1.scatter(spikes, y, c=states_colors[_aux_c], s=1.0, marker = '|')

            _aux_c += 1

        for _s, neur_IDs in self.sFSA_sim_data['sFSA']['I'].items():

            for neuron_id in neur_IDs:
                if neuron_id in E_spk_trains.keys():
                    spikes = E_spk_trains[neuron_id]
                    y = np.ones_like(spikes) * neuron_id
                    ax1.scatter(spikes, y, c='k', s=1.0, marker = '|')

        # - Axis for the voltage threshold traces.

        _aux_c = 0
        for state, Vth in self.sFSA_sim_data['Vth_traces'].items():

            Vth_mu = np.mean(Vth, axis = 0)
            Vth_sg = np.std(Vth, axis = 0)

            ax2.plot(self.sFSA_sim_data['sim_t'], Vth_mu, color = states_colors[_aux_c])
            ax2.plot(self.sFSA_sim_data['sim_t'], np.max(Vth, axis = 0), color = states_colors[_aux_c], ls = '--', lw = 0.8)
            ax2.plot(self.sFSA_sim_data['sim_t'], np.min(Vth, axis = 0), color = states_colors[_aux_c], ls = '--', lw = 0.8)
            ax2.fill_between(self.sFSA_sim_data['sim_t'], Vth_mu-Vth_sg, Vth_mu+Vth_sg, alpha = 0.2, color = states_colors[_aux_c])

            _aux_c += 1

        ax2.hlines(y=self.sFSA_sim_data['thr_GO_state'], xmin = 0, xmax = int(round(self.sFSA_sim_data['sim_t'][-1]))+1, colors = 'b', linestyles = 'dashed', lw = 0.75, ls = '-.')

        # - Set the axis labels and title.

        ax1.set_ylabel('Neuron ID')
        ax1.set_yticks(np.arange(0, self.sFSA_sim_data['sFSA']['N_e'], 64))
        ax1.set_ylim(0, self.sFSA_sim_data['sFSA']['N_e'])
        ax1.set_xticks(np.arange(0, int(round(self.sFSA_sim_data['sim_t'][-1]))+0.5, 0.5))
        ax1.set_xlim(0, int(round(self.sFSA_sim_data['sim_t'][-1])))

        ax2.set_ylabel(r'$V_{th}$')
        ax2.set_xlabel('time [s]')
        ax2.set_xticks(np.arange(0, int(round(self.sFSA_sim_data['sim_t'][-1]))+0.5, 0.5))

        plt.tight_layout()

        plt.savefig(os.path.join(self.data_folder, f'network_activity{name_ext}.pdf'))

        plt.close()

    def makeSimulationFolder(self, sub_dir = ''):

        # - Creating data folder.

        if self.data_folder == '':

            if sub_dir != '':
                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', sub_dir))

                if not (os.path.isdir(data_folder)):
                    os.mkdir(data_folder)

                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', sub_dir, f'sFSA_{self.sFSA_id}'))
            else:
                data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', f'sFSA_{self.sFSA_id}'))

            if not (os.path.isdir(data_folder)):
                os.mkdir(data_folder)

            self.data_folder = data_folder

    def storeSfsa(self):
        '''
        Stores the state of the RCN (brian2's Network.store) implementing the FSA.
        '''

        # - Creating data folder.

        if self.data_folder == '':

            data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', f'sFSA_{self.sFSA_id}'))

            if not (os.path.isdir(data_folder)):
                os.mkdir(data_folder)

            self.data_folder = data_folder

        # - Storing RCN initial state.
        self.__RCN.net.store(name = f'{self.sFSA_id}_initial_state', filename = os.path.join(self.data_folder, f'{self.sFSA_id}_initial_state'))

    def restoreSfsa(self):
        '''
        Restores the state of the RCN (brian2's Network.store) implementing the FSA.
        '''

        self.__RCN.net.restore(name = f'{self.sFSA_id}_initial_state', filename = os.path.join(self.data_folder, f'{self.sFSA_id}_initial_state'))



