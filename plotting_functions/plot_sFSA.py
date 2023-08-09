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

for folder in folders:

    print(f'> folder: {folder}')

    filename = '//sSFA_sim_data.pickle'
    filepath = f'D://A_PhD//GitHub//wm_colaboration//results//RCN_FSA//{folder}//BA_15_GS_100_W_7_I_20'

    with open(filepath + filename, 'rb') as file:
        data = pickle.load(file)

    E_spk_trains = data['E_spk_trains']

    # Create a figure and axis for the raster plot
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

    if len(data['i_tokens_twindows']) != 0:

        _inp_seq = [i for i in data['input_sequence'] if i not in [__s for __s, neur_IDs in data['sFSA']['S'].items()]]

        for i in range(len(_inp_seq)):

            _s = _inp_seq[i]

            _x = np.round((data['i_tokens_twindows'][i][1]+data['i_tokens_twindows'][i][0])/2, 1)
            _y = np.round((data['sFSA']['I'][_s][-1]+data['sFSA']['I'][_s][0])/2, 1)

            _size = 30

            ax1.text(
                _x, 
                _y-15, 
                _s,
                color = 'w',
                fontsize = _size)

    # Set the axis labels and title
    ax1.set_ylabel('Neuron ID')
    ax1.set_xlabel('time [s]')
    ax1.set_yticks(np.arange(0, data['sFSA']['N_e'], 64))
    ax1.set_ylim(0, data['sFSA']['N_e'])

    ax1.set_xlim(0, round(data['sim_t'][-1]))



    plt.tight_layout()

    plt.show()
