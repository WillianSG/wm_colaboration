import brian2.monitors
import numpy as np


def has_spiked(window, monitor):
    assert isinstance(monitor, brian2.monitors.SpikeMonitor)

    has_spiked = {}
    spikes = monitor.spike_trains()
    for i, spks in spikes.items():
        sp = spks[(spks > window[0]) & (spks < window[1])]
        # print(i, np.count_nonzero(sp))
        has_spiked[i] = bool(np.count_nonzero(sp))

    return has_spiked
