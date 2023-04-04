import pickle
import numpy as np
import matplotlib.pyplot as plt

folder = '04-04-2023_16-18-44'
filename = '//RCN_attractors_data.pickle'
filepath = f'D://A_PhD//GitHub//wm_colaboration//results//RCN_state_transition//{folder}//BA_15_GS_100_W_10_I_20'

with open(filepath + filename, 'rb') as file:
    data = pickle.load(file)

E_spk_trains = data['E_spk_trains']

# Create a figure and axis for the raster plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10,5))

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

ax2.hlines(y=-data['thr_GO_state'], xmin=data['sim_t'][0], xmax=data['sim_t'][-1], colors='k', linestyles='dashed', lw = 0.5)

for i in range(100):

    ax3.plot(data['sim_t'], data['E_w_interattra'][i]*1000, lw = 0.5)

ax3.set_ylabel('w_ef [mv]')

# ax1.legend(loc = 'best', framealpha = 0.0)
# ax2.legend(loc = 'best', framealpha = 0.0)
# ax3.legend(loc = 'best', framealpha = 0.0)

# Show the plot
plt.show()