# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence
"""
import sys, os, pickle
from brian2 import Hz, mV, second, prefs
from datetime import datetime

prefs.codegen.target = 'numpy'

if sys.platform in ['linux', 'win32']:
    sys.path.append(os.path.abspath(os.path.join(__file__ , '../../')))
    from recurrent_competitive_network import RecurrentCompetitiveNet
else:
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet

class sFSA:
    def __init__(self, FSA_model: dict, args):

        self.sFSA_id = str(datetime.utcnow()).replace(' ', '_').replace(':', '-').replace('.', '')

        self.plasticity_rule = 'LR4'
        self.parameter_set = '2.2'
        self.attractor_size = 64

        self.__FSA_model = FSA_model
        self.__args = args

        self.__RCN = RecurrentCompetitiveNet(self.plasticity_rule, self.parameter_set)

        self.__loadSfsaArgs()
        self.__loadSfsaModel()
        self.__enterStartState()

        self.inputed_sequence = []

    # - Private Functions -

    def __loadSfsaArgs(self):

            self.__RCN.thr_GO_state = self.__args.thr_GO_state
            self.__RCN.E_E_syn_matrix_snapshot = False
            self.__RCN.w_e_i = 3 * mV                      
            self.__RCN.w_max = 10 * mV                     
            self.__RCN.spont_rate = self.__args.ba_rate * Hz
            self.__RCN.w_e_i = 3 * mV                      
            self.__RCN.w_i_e = self.__args.W_ie * mV              
            self.__RCN.tau_Vth_e = 0.1 * second            
            self.__RCN.k = 4    
            self.__RCN.U = 0.2

    def __loadSfsaModel(self):
        
        self.__sfsa_state = {'S': {}, 'I': {}, 'T': {}, 'N_e': 0, 'N_i': 0}

        # - Computing required RCN size (# of attractors) to implement SFA model.

        _new_state_trans_count = 0
        for transition in self.__FSA_model['T']:
            
            _s0 = transition.split('->')[0].split(', ')[0].replace('(', '')     # current state.
            _i = transition.split('->')[0].split(', ')[1].replace(')', '')      # input token.
            _s1 = transition.split('->')[1]

            if _s0 != _s1:
                _new_state_trans_count += 1

        self.__sfsa_state['N_e'] = int((len(self.__FSA_model['S']) + len(self.__FSA_model['I']) + _new_state_trans_count)*self.attractor_size)
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


        for transition in self.__FSA_model['T']:

            # - Parsing transition.
            
            _s0 = transition.split('->')[0].split(', ')[0].replace('(', '')     # current state.
            _i = transition.split('->')[0].split(', ')[1].replace(')', '')      # input token.
            _s1 = transition.split('->')[1]

            if _s0 != _s1:
                # - Transition to a different state.

                self.__sfsa_state['S'][f'{_s0}-{_s1}'] = list(range(ids_start, ids_end))

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

        for tid, transition in self.__sfsa_state['T'].items():

            # - Parsing transition.
            
            _s0 = transition.split('->')[0].split(', ')[0].replace('(', '')     # current state.
            _i = transition.split('->')[0].split(', ')[1].replace(')', '')      # input token.
            _s1 = transition.split('->')[1]                                     # next state.

            # - Setting connections for state->input->state transitions (transfering cue).
            
            self.__RCN.set_synapses_A_2_B(A_ids = self.__sfsa_state['I'][_i], B_ids = self.__sfsa_state['S'][_s0], weight = self.__args.w_trans*mV)

            # - Setting connections for state-input matching leading to next state.

            if self.__sfsa_state['S'][_s0] != self.__sfsa_state['S'][_s1]:

                self.__RCN.set_synapses_A_2_GO(A_ids = self.__sfsa_state['S'][_s0], GO_ids = self.__sfsa_state['S'][f'{_s0}-{_s1}'], weight = self.__args.w_acpt*mV)
                self.__RCN.set_synapses_A_2_GO(A_ids = self.__sfsa_state['I'][_i], GO_ids = self.__sfsa_state['S'][f'{_s0}-{_s1}'], weight = self.__args.w_acpt*mV)

                self.__RCN.set_synapses_A_2_B_trans(A_ids = self.__sfsa_state['S'][f'{_s0}-{_s1}'], B_ids = self.__sfsa_state['S'][_s1], weight = (self.__args.w_trans*3)*mV)

    def __enterStartState(self):

        # - Setting/sending input to attractor.

        start_state = self.__FSA_model['S'][0]

        act_ids = self.__RCN.generic_stimulus(
                frequency = 10000*Hz, 
                stim_perc = self.__args.gs_percent, 
                subset = self.__sfsa_state['S'][start_state])

        self.__RCN.run_net(duration = 0.2)

        self.__RCN.generic_stimulus_off(act_ids)

        # - Running network without external input (attractor evolution after stimulation).

        self.__RCN.run_net(duration = 0.4)

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
                frequency = 10000*Hz, 
                stim_perc = self.__args.gs_percent, 
                subset = self.__sfsa_state['I'][input_token])
        
        self.__RCN.run_net(duration = 0.6)

        self.__RCN.generic_stimulus_off(act_ids)

        # - Running network without external input (attractor evolution after stimulation).

        self.__RCN.run_net(duration = 5.0)

        # @TODO return sequence of activations from network activity.

    # - Public Functions -

    def feedInputWord(self, input_word):

        for token in input_word:

            self.__feedInputToken(token)

        self.__RCN.run_net(duration = 1.0)

    def exportSfsaData(self):

        # - Creating data folder.

        data_folder = os.path.abspath(os.path.join(__file__ , '../../../', 'results', f'sFSA_{self.sFSA_id}'))

        if not (os.path.isdir(data_folder)):
            os.mkdir(data_folder)

        # - Export metadata.

        with open(os.path.join(data_folder, 'parameters.txt'), "w") as file:
            for arg in vars(self.__args):
                arg_value = getattr(self.__args, arg)
                if arg_value is not None:
                    file.write(f"\n{arg}: {arg_value}\n")

        # - Export data.

        _E_spk_trains = {key: value for key, value in self.__RCN.E_mon.spike_trains().items()}          # excitatory spike trains.

        sim_t = self.__RCN.E_rec.t/second

        fn = os.path.join(
            data_folder,
            'sSFA_simulation_data.pickle')

        with open(fn, 'wb') as f:
            pickle.dump({
                'E_spk_trains': _E_spk_trains,
                'sim_t': sim_t,
                'sFSA': self.__sfsa_state,
                'input_sequence': self.inputed_sequence
            }, f)

        return data_folder
    
    def plotSfsaNetwork(self, data_path):
        import matplotlib.pyplot as plt
        import numpy as np

        # - Load data.

        with open(os.path.join(data_path, 'sSFA_simulation_data.pickle'), 'rb') as file:
            data = pickle.load(file)

        E_spk_trains = data['E_spk_trains']

        # - Create a figure and axis for the raster plot.

        fig, ax1 = plt.subplots(nrows = 1, sharex = True, figsize = (15, 5))

        states_colors = ['mediumslateblue', 'magenta', 'darkorange', 'purple', 'b', 'r', 'g', 'darkred']

        _aux_c = 0

        for _s, neur_IDs in data['sFSA']['S'].items():

            for neuron_id in neur_IDs:
                if neuron_id in E_spk_trains.keys():
                    spikes = E_spk_trains[neuron_id]
                    y = np.ones_like(spikes) * neuron_id
                    ax1.scatter(spikes, y, c=states_colors[_aux_c], s=1.0, marker = '|')

            _aux_c += 1

        for _s, neur_IDs in data['sFSA']['I'].items():

            for neuron_id in neur_IDs:
                if neuron_id in E_spk_trains.keys():
                    spikes = E_spk_trains[neuron_id]
                    y = np.ones_like(spikes) * neuron_id
                    ax1.scatter(spikes, y, c='k', s=1.0, marker = '|')

        # - Set the axis labels and title.

        ax1.set_ylabel('Neuron ID')
        ax1.set_xlabel('time [s]')
        ax1.set_yticks(np.arange(0, data['sFSA']['N_e'], 64))
        ax1.set_ylim(0, data['sFSA']['N_e'])

        ax1.set_xticks(np.arange(0, int(round(data['sim_t'][-1]))+1, 1))

        ax1.set_xlim(0, int(round(data['sim_t'][-1])))

        plt.tight_layout()

        plt.show()
        plt.close()


