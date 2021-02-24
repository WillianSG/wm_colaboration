import setuptools
from brian2 import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

start_scope()

P = PoissonGroup(10, rates = 1*Hz)
P_mon = SpikeMonitor(P)

net = Network(P, P_mon)
net.run(2*second)

# S = SpikeGeneratorGroup(N = 10,
# 			indices = np.arange(10),
# 			times = abs(np.random.randn(10))*1*ms,
# 			name = 'S')

# S_mon = SpikeMonitor(S)

plt.plot(P_mon.t, P_mon.i, '.k')
plt.title('Activity')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()