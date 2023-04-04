import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = '//RCN_attractors_data.pickle'
filepath = 'D://A_PhD//GitHub//wm_colaboration//results//RCN_state_transition//04-04-2023_14-18-37//BA_15_GS_100_W_10_I_20'

with open(filepath + filename, 'rb') as file:
    data = pickle.load(file)

E_spk_trains = data['E_spk_trains']

# Specify the neuron lists A and B
list_A = list(range(0, 64))
list_B = list(range(100, 164))

# Create a figure and axis for the raster plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(10,5))

# Plot the spikes for neurons in list A in blue
for neuron_id in list_A:
    if neuron_id in E_spk_trains.keys():
        spikes = E_spk_trains[neuron_id]
        y = np.ones_like(spikes) * neuron_id
        ax1.scatter(spikes, y, c='b', s=5, label = 'A1')

# Plot the spikes for neurons in list B in red
for neuron_id in list_B:
    if neuron_id in E_spk_trains.keys():
        spikes = E_spk_trains[neuron_id]
        y = np.ones_like(spikes) * neuron_id
        ax1.scatter(spikes, y, c='r', s=5, label = 'A2')

# Set the axis labels and title
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Neuron ID')

ax2.plot(data['sim_t'], np.mean(data['A1_Vth']*1000, axis = 0), color = 'b', label = 'A1')
ax2.plot(data['sim_t'], np.mean(data['A2_Vth']*1000, axis = 0), color = 'r', label = 'A2')
ax2.set_ylabel('Vth [mv]')

for i in range(100):

    ax3.plot(data['sim_t'], data['E_w_interattra'][i]*1000, lw = 0.5)

ax3.set_ylabel('w_ef [mv]')

# ax1.legend(loc = 'best', framealpha = 0.0)
# ax2.legend(loc = 'best', framealpha = 0.0)
# ax3.legend(loc = 'best', framealpha = 0.0)

# Show the plot
plt.show()