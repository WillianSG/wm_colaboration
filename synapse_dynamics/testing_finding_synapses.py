import numpy as np
import matplotlib.pyplot as plt

from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor
from brian2 import ms, Hz
from brian2 import run

from helper_functions.other import *

np.random.seed(7)

eqs = '''
dv/dt = (1-v)/tau : 1
tau : second
'''
G = NeuronGroup(4, eqs, threshold='v>0.5', reset='v = 0', method='exact')
G.tau = 10 * ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G,
             '''
             w:1
             ''', on_pre='v_post += 0.2')
S.connect('i!=j', p=0.4)
S.w = 'i'

M = StateMonitor(G, 'v', record=True)
# syn_mon = StateMonitor(S, 'w', record=True)

run(100 * ms)

# plt.plot(M.t / ms, M.v[0], label='Neuron 0')
# plt.plot(M.t / ms, M.v[1], label='Neuron 1')
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.legend()
# plt.show()

visualise_connectivity(S)

print(S.i)
print(S.j)
for i, j in zip(S.i, S.j):
    print(i, '→', j)
print('Synapses efferent from 2 to 4\n\t', S[2:4, :])
print('Synapses afferent to 2 to 4\n\t', S[:, 2:4])
print('Value of w for synapses efferent from neurons 2 to 4\n\t', S.w[2:4, :])
print('Value of w for synapses afferent to neurons 2 to 4\n\t', S.w[:, 2:4])
