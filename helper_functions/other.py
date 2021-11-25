import brian2.monitors
import numpy as np
import matplotlib.pyplot as plt

"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Cognitive Modelling

Function:
- Has neuron spiked in the supplied time window?

Script arguments:
-

Script output:
- has_spiked: dictionary with neurons as keys and True or False depending if the neuron has spiked or not within the time window

Comments:
- 

Inputs:
- window(tuple, list, Numpy array, or int, float): 2-element list-like object containing endpoints of time window, if not list-like then the lower bound is assumed t=0
- monitor(brian2.monitor.SpikeMonitor): Brian2 SpikeMonitor object with recorded spikes
"""


def has_spiked(window, monitor):
    assert isinstance(monitor, brian2.monitors.SpikeMonitor)
    assert isinstance(window, brian2.Quantity)
    assert window.size == 1 or window.size == 2
    if window.size == 1:
        window = (0, window)

    spikes = monitor.spike_trains()
    has_spiked = np.zeros(len(spikes), dtype=bool)
    for i, spks in spikes.items():
        sp = spks[(spks > window[0]) & (spks < window[1])]
        # print(i, np.count_nonzero(sp))
        has_spiked[i] = bool(np.count_nonzero(sp))

    return has_spiked


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.show()
