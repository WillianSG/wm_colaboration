import numpy as np

def save_spikes_pyspike(spikes):
    with open('../graph_analysis/E_spikes.txt', 'w') as file:
        file.write('# Spikes from excitatory population')
        for v in spikes.values():
            np.savetxt(file, np.array(v), newline=' ')
            file.write('\n')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pyspike as spk

    spike_trains = spk.load_spike_trains_from_txt("../graph_analysis/E_spikes.txt",edges=(0, 6), ignore_empty_lines=False)
    # plt.plot(spike_trains)
    spike_profile_A1 = spk.spike_profile(spike_trains, indices=(0,64))
    spike_profile_A2 = spk.spike_profile(spike_trains, indices=(100,164))
    x, y = spike_profile_A1.get_plottable_data()
    plt.plot(x, y, '--', label='A1')
    x, y = spike_profile_A2.get_plottable_data()
    plt.plot(x, y, '--', label='A2')
    plt.legend()
    plt.ylim(0,1)
    # print("SPIKE distance: %.8f" % spike_profile.avrg())
    plt.show()