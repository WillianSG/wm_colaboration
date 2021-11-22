import numpy as np
import matplotlib.pyplot as plt
import math

from brian2 import PoissonGroup, NeuronGroup, Synapses, SpikeMonitor, TimedArray
from brian2 import ms, Hz
from brian2 import run

from helper_functions.load_stimulus import load_stimulus

n_neurons = 256

offset = [0, 100]
rates_array = []

for off in offset:
    stim = load_stimulus('flat_to_E_fixed_size', stimulus_size=64, offset=off)
    rate = np.zeros(n_neurons)
    rate[stim] = 6600 * Hz

    img_size = int(math.sqrt(n_neurons))
    img = np.zeros((img_size, img_size))
    stim_index = np.unravel_index(stim, img.shape)

    rates_array.append(rate)

rates_array = TimedArray(rates_array * Hz, dt=100 * ms)

P = PoissonGroup(n_neurons, rates='rates_array(t,i)')
G = NeuronGroup(n_neurons, 'dv/dt = -v / (10*ms) : 1', threshold='v>0.1', reset='v=0')
S = Synapses(P, G, on_pre='v+=0.1')
S.connect(j='i')

spikemon_P = SpikeMonitor(P)
spikemon_G = SpikeMonitor(G)

run(200 * ms)

fig, axes = plt.subplots(2, 1)
for ax in axes:
    ax.set_ylim(0, n_neurons)
    ax.set_ylabel('Neuron ID')
axes[0].plot(spikemon_P.t / ms, spikemon_P.i, '.')
axes[1].plot(spikemon_G.t / ms, spikemon_G.i, '.')
axes[1].set_xlabel('Time (s)')
plt.show()
