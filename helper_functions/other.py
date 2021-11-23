import brian2.monitors
import numpy as np

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

    has_spiked = {}
    spikes = monitor.spike_trains()
    for i, spks in spikes.items():
        sp = spks[(spks > window[0]) & (spks < window[1])]
        # print(i, np.count_nonzero(sp))
        has_spiked[i] = bool(np.count_nonzero(sp))

    return has_spiked
