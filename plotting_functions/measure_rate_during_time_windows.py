import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../')))

    sys.path.append(os.path.join(root, 'helper_functions'))

    from firing_rate_histograms import firing_rate_histograms

# folder = str(sys.argv[1])
folders = os.listdir('D://A_PhD//GitHub//wm_colaboration//results//RCN_FSA')

def get_spikes_in_twindow(spikes, t_window):

    return [i for i in list(spikes/second) if i >= t_window[0] and i <= t_window[1]]

def get_attractor_mean_rate(spikes, _neur_IDS):

    [t_hist_count,
    t_hist_edgs,
    t_hist_bin_widths,
    t_hist_fr] = firing_rate_histograms(
        tpoints=np.array(spikes)*second,
        inds=list(_neur_IDS),
        bin_width=50*ms,
        N_pop=len(_neur_IDS),
        flag_hist='time_resolved' )

    return np.round(np.mean(t_hist_fr), 1)

def extract_parameters(file_path, output_path, CR):
    parameters = {}  # Dictionary to store the extracted parameters

    # Specify the lines to extract from the file
    lines_to_extract = ['ba_rate', 'W_ie', 'inh_rate', 'w_acpt', 'w_trans']

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            key, value = line.split(':')
            key = key.strip()
            value = float(value.strip())

            if key in lines_to_extract:
                parameters[key] = value

    parameters['CR'] = CR

    # Export the dictionary to a pickle file
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)


if not os.path.exists('D://A_PhD//GitHub//wm_colaboration//results//sFSA_gauss_search'):
        os.makedirs('D://A_PhD//GitHub//wm_colaboration//results//sFSA_gauss_search')

for folder in folders:

    _sub_folder = os.listdir(f'D://A_PhD//GitHub//wm_colaboration//results//RCN_FSA//{folder}')[0]

    filename = '//sSFA_sim_data.pickle'
    filepath = f'D://A_PhD//GitHub//wm_colaboration//results//RCN_FSA//{folder}//{_sub_folder}'

    if len(os.listdir(filepath)) > 0:

        with open(filepath + filename, 'rb') as file:
            data = pickle.load(file)

        E_spk_trains = data['E_spk_trains']

        # # Create a figure and axis for the raster plot
        # fig, ax1 = plt.subplots(nrows=1, sharex=True, figsize=(15, 5))

        # states_colors = ['mediumslateblue', 'magenta', 'darkorange', 'purple', 'b', 'r', 'g', 'darkred']

        # _aux_c = 0

        # for _s, neur_IDs in data['sFSA']['S'].items():

        #     for neuron_id in neur_IDs:
        #         if neuron_id in E_spk_trains.keys():
        #             spikes = E_spk_trains[neuron_id]
        #             y = np.ones_like(spikes) * neuron_id
        #             ax1.scatter(spikes, y, c=states_colors[_aux_c], s=1.0, marker = '|')

        #     _aux_c += 1

        # for _s, neur_IDs in data['sFSA']['I'].items():

        #     for neuron_id in neur_IDs:
        #         if neuron_id in E_spk_trains.keys():
        #             spikes = E_spk_trains[neuron_id]
        #             y = np.ones_like(spikes) * neuron_id
        #             ax1.scatter(spikes, y, c='k', s=1.0, marker = '|')

        # if len(data['i_tokens_twindows']) != 0:

        #     _inp_seq = [i for i in data['input_sequence'] if i not in [__s for __s, neur_IDs in data['sFSA']['S'].items()]]

        #     for i in range(len(_inp_seq)):

        #         _s = _inp_seq[i]

        #         _x = np.round((data['i_tokens_twindows'][i][1]+data['i_tokens_twindows'][i][0])/2, 1)
        #         _y = np.round((data['sFSA']['I'][_s][-1]+data['sFSA']['I'][_s][0])/2, 1)

        #         _size = 30

        #         ax1.text(
        #             _x, 
        #             _y-15, 
        #             _s,
        #             color = 'w',
        #             fontsize = _size)

        # =======================================================================================================================

        def check_is_active(mean_rate):

            if mean_rate >= 7:

                return True

            else:

                return False

        _margin_1 = 0.1
        _margin_2 = 0.25

        _states_correct_sequence = {}
        _state_sequence_list = ['I_B_1', 'II_A_1', 'II_B_1', 'I_A_2', 'I_B_2', 'III_A', 'III_B', 'II_A_2']

        _state_sequence_list_twin = []

        for i in range(1, len(data['input_sequence'])):

            _currentstate_start = np.round(data['i_tokens_twindows'][i-1][0] + data['stimulation_time_windows'][2] + _margin_1, 2)
            _currentstate_end = np.round(_currentstate_start + 2.0 - _margin_2, 2)

            _state_sequence_list_twin.append((_currentstate_start, _currentstate_end))

            # _nextstate_start = np.round(_currentstate_end + _margin_2, 2)
            _nextstate_end = np.round(_currentstate_end + data['stimulation_time_windows'][3] - 1.5 - _margin_2, 2)
            _nextstate_start = np.round(_nextstate_end - 0.5, 2)

            _state_sequence_list_twin.append((_nextstate_start, _nextstate_end))

            # ax1.axvline(x=_currentstate_start, color='k', linestyle='--')
            # ax1.axvline(x=_currentstate_end, color='k', linestyle='--')

            # ax1.axvline(x=_nextstate_start, color='g', linestyle='-', alpha = 0.25)
            # ax1.axvline(x=_nextstate_end, color='g', linestyle='-', alpha = 0.25)

        for i in range(len(_state_sequence_list)):

            _states_correct_sequence[_state_sequence_list_twin[i]] = _state_sequence_list[i].split('_')[0]

        _twindow_activestate = {}

        for t_win in _state_sequence_list_twin:

            _twindow_activestate[t_win] = []

            for _s, neur_IDs in data['sFSA']['S'].items():

                spikes = []

                for neuron_id in neur_IDs:
                    if neuron_id in E_spk_trains.keys():
                        _ = get_spikes_in_twindow(E_spk_trains[neuron_id], t_win)
                        spikes += list(_)

                t_win_rate = get_attractor_mean_rate(spikes, neur_IDs)

                if check_is_active(t_win_rate):

                    _twindow_activestate[t_win].append(_s)


                # print(_s, t_win, t_win_rate)

        _correct = 0
        _wrong = 0

        for key, val in _twindow_activestate.items():

            if len(_twindow_activestate[key]) == 1 and _twindow_activestate[key][0] == _states_correct_sequence[key]:

                _correct += 1

            else:

                _wrong += 1

            # print(key, val)


        _CR = _correct/(_correct+_wrong)

        if _CR >= 0.8:

            print(f'> folder: {folder}, {_sub_folder} | correctness: {_CR} ({_correct}, {_wrong})')

            extract_parameters(
                filepath+'//parameters.txt', 
                f'D://A_PhD//GitHub//wm_colaboration//results//sFSA_gauss_search//{folder}_params_CR.pickle',
                _CR)

        # =======================================================================================================================

        # Set the axis labels and title
        # ax1.set_ylabel('Neuron ID')
        # ax1.set_xlabel('time [s]')
        # ax1.set_yticks(np.arange(0, data['sFSA']['N_e'], 64))
        # ax1.set_ylim(0, data['sFSA']['N_e'])

        # ax1.set_xlim(0, round(data['sim_t'][-1]))

        # plt.tight_layout()

        # plt.show()
