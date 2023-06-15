import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../')))

    sys.path.append(os.path.join(root, 'helper_functions'))

    from firing_rate_histograms import firing_rate_histograms

folder = str(sys.argv[1])
filename = '//RCN_attractors_data.pickle'
filepath = f'D://A_PhD//GitHub//wm_colaboration//results//RCN_state_transition//{folder}//BA_15_GS_100_W_10_I_20'

with open(filepath + filename, 'rb') as file:
    data = pickle.load(file)

E_spk_trains = data['E_spk_trains']

# Create a figure and axis for the raster plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(10,7.5))

# Plot the spikes for neurons in list A in blue
for neuron_id in data['attractor_A']:
    if neuron_id in E_spk_trains.keys():
        spikes = E_spk_trains[neuron_id]
        y = np.ones_like(spikes) * neuron_id
        ax1.scatter(spikes, y, c='b', s=5, label = 'A1')

# Plot the spikes for neurons in list B in red
for neuron_id in data['attractor_B']:
    if neuron_id in E_spk_trains.keys():
        spikes = E_spk_trains[neuron_id]
        y = np.ones_like(spikes) * neuron_id
        ax1.scatter(spikes, y, c='r', s=5, label = 'A2')

# Plot the spikes for neurons in list C in red
for neuron_id in data['attractor_C']:
    if neuron_id in E_spk_trains.keys():
        spikes = E_spk_trains[neuron_id]
        y = np.ones_like(spikes) * neuron_id
        ax1.scatter(spikes, y, c='g', s=5, label = 'A3')

# Set the axis labels and title
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Neuron ID')

A1_twindow = data['A_t_window']
A2_twindow = data['B_t_window']

ax1.axvline(
    x = A2_twindow[0] + A1_twindow[2] + data['delay_A2GO']/second,
    color = 'r', lw = 1.0, ls = '--')

ax1.axvline(
    x = A2_twindow[0] + A1_twindow[2] + data['delay_A2B']/second + data['delay_A2GO']/second,
    color = 'b', lw = 1.25, ls = '--')

A1_mu = np.mean(data['A1_Vth']*1000, axis = 0)
A1_sg = np.std(data['A1_Vth']*1000, axis = 0)

A2_mu = np.mean(data['A2_Vth']*1000, axis = 0)
A2_sg = np.std(data['A2_Vth']*1000, axis = 0)

A3_mu = np.mean(data['A3_Vth']*1000, axis = 0)
A3_sg = np.std(data['A3_Vth']*1000, axis = 0)

ax2.plot(data['sim_t'], A1_mu, color = 'b', label = 'A1')
ax2.fill_between(data['sim_t'], A1_mu-A1_sg, A1_mu+A1_sg, alpha=0.2, color='b')
                 
ax2.plot(data['sim_t'], np.mean(data['A2_Vth']*1000, axis = 0), color = 'r', label = 'A2')
ax2.fill_between(data['sim_t'], A2_mu-A2_sg, A2_mu+A2_sg, alpha=0.2, color='r')

ax2.plot(data['sim_t'], np.mean(data['A3_Vth']*1000, axis = 0), color = 'g', label = 'A3')
ax2.fill_between(data['sim_t'], A3_mu-A3_sg, A3_mu+A3_sg, alpha=0.2, color='g')

ax2.set_ylabel('Vth [mv]')

ax2.hlines(y=data['thr_GO_state'], xmin=data['sim_t'][0], xmax=data['sim_t'][-1], colors='k', linestyles='dashed', lw = 0.5)

for i in range(len(data['E_w_interattra'])):

    ax3.plot(data['sim_t'], data['E_w_interattra'][i]*1000, lw = 0.25)

ax2.axvline(
    x = A2_twindow[0] + A1_twindow[2] + data['delay_A2GO']/second,
    color = 'r', lw = 1.0, ls = '--')

ax2.axvline(
    x = A2_twindow[0] + A1_twindow[2] + data['delay_A2B']/second + data['delay_A2GO']/second,
    color = 'b', lw = 1.25, ls = '--')

ax3.set_ylabel('w_ef [mv]')

i_spks = []
for i_id, i_spkt in data['I_spk_trains'].items():

    i_spks += list(i_spkt/second)

[input_e_t_hist_count,
    input_e_t_hist_edgs,
    input_e_t_hist_bin_widths,
    input_e_t_hist_fr ] = firing_rate_histograms(
        tpoints=np.array(i_spks)*second,
        inds=list(range(0, 64)),
        bin_width=100*ms,
        N_pop=len(list(range(0, 64))),
        flag_hist='time_resolved' )

ax4.bar( input_e_t_hist_edgs[ :-1 ], input_e_t_hist_fr,
    input_e_t_hist_bin_widths,
    edgecolor='k',
    color='purple',
    linewidth=1.0 )

ax4.set_ylabel('inh pop [Hz]')

ax1.set_xlim(0, round(data['sim_t'][-1]))
ax2.set_xlim(0, round(data['sim_t'][-1]))
ax3.set_xlim(0, round(data['sim_t'][-1]))
ax4.set_xlim(0, round(data['sim_t'][-1]))

plt.show()
