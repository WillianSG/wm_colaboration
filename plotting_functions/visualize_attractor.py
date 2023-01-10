from brian2 import mV, second
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def visualize_attractor(attractor, network):

	attractor.sort()

	fig = plt.figure(constrained_layout = True, figsize = (10, 7.5))

	widths = [5, 5]
	heights = [2.5, 2.5, 2.5]

	spec = gridspec.GridSpec(
		ncols = len(widths), 
		nrows = len(heights), 
		width_ratios = widths,
		height_ratios = heights,
		figure = fig)

	# --- 1st column --------------------------------------------------------------------

	ax1 = fig.add_subplot(spec[0, 0])

	yticks = list(np.arange(-50, -70, -2))
	yticks.reverse()

	mean_Vmem = np.mean(network.E_rec.Vm[attractor, :], axis = 0)/mV
	std_Vmem = np.std(network.E_rec.Vm[attractor, :], axis = 0)/mV

	plt.plot(network.E_rec.t/second, mean_Vmem, lw = 1.5, c = 'b')

	plt.fill_between(
		network.E_rec.t/second, 
		mean_Vmem-std_Vmem, 
		mean_Vmem+std_Vmem,
		alpha = 0.3,
		color = 'b')

	ax1.set_ylabel(r'$V_{mem}$' + ' [mV]', color = 'b')
	ax1.set_xlabel('time [s]')
	ax1.set_yticks(yticks)
	ax1.tick_params(axis = 'y', labelcolor = 'b')

	ax1a = ax1.twinx()

	mean_Vth = np.mean(network.E_rec.Vth_e[attractor, :], axis = 0)/mV
	std_Vth = np.std(network.E_rec.Vth_e[attractor, :], axis = 0)/mV

	plt.plot(network.E_rec.t/second, mean_Vth, lw = 1.5, c = 'r', alpha = 0.5)

	plt.fill_between(
		network.E_rec.t/second, 
		mean_Vth-std_Vth, 
		mean_Vth+std_Vth,
		alpha = 0.2,
		color = 'r')

	ax1a.set_ylabel(r'$V_{th}$' + ' [mV]', color = 'r')
	ax1a.set_yticks(yticks)
	ax1a.tick_params(axis = 'y', labelcolor = 'r')

	ax2 = fig.add_subplot(spec[1, 0])

	spk_mon_ids, spk_mon_ts = network.get_spks_from_pattern_neurons()

	plt.plot(spk_mon_ts, spk_mon_ids, '|', color = 'k', zorder = 0)

	ax2.set_ylabel(r'$N_{ID}$')
	ax2.set_xlabel('time [s]')
	ax2.set_yticks(np.arange(0, attractor[-1], 10))

	ax3 = fig.add_subplot(spec[2, 0])

	xs_, t_array, tau_d = network.get_x_traces_from_pattern_neurons()
	us_, t_array, U, tau_f = network.get_u_traces_from_pattern_neurons()

	all_x = []

	for item in xs_:

		all_x.append(item['x'])

	all_u = []

	for item in us_:

		all_u.append(item['u'])

	x_mean = np.array(np.mean(all_x, axis = 0))
	u_mean = np.array(np.mean(all_u, axis = 0))
	_Vepsp = (x_mean*u_mean)/network.U

	plt.plot(t_array, x_mean, lw = 1.5, color = 'g', label = 'x')
	plt.plot(t_array, u_mean, lw = 1.5, color = 'brown', label = 'u')

	ax3.set_xlabel('time [s]')
	ax3.set_yticks(np.arange(0.0, 1.2, 0.2))
	ax3.set_ylim(0.0, 1.0)

	plt.legend(loc = 'best', framealpha = 0)

	ax3a = ax3.twinx()

	plt.plot(t_array, _Vepsp, lw = 1.5, color = 'purple', ls = '--')
	
	ax3a.set_yticks(np.arange(0.0, 4.5, 0.5))
	ax3a.set_ylim(0.0, 4.0)

	ax3a.set_ylabel(r'$V_{epsp}$' + ' [mV]', color = 'purple')

	ax3a.tick_params(axis = 'y', labelcolor = 'purple')

	# --- 2nd column --------------------------------------------------------------------

	ax4 = fig.add_subplot(spec[1, 1])

	plt.plot(network.E_rec.t/second, network.E_rec.Vm[attractor[0]]/mV, lw = 1.5, c = 'b')

	ax4.set_ylabel(r'$V_{mem}$' + ' [mV]', color = 'b')
	ax4.set_xlabel('time [s]')
	ax4.set_yticks(yticks)
	ax4.tick_params(axis = 'y', labelcolor = 'b')

	ax4a = ax4.twinx()

	plt.plot(network.E_rec.t/second, network.E_rec.Vth_e[attractor[0]]/mV, lw = 1.5, c = 'r')

	ax4a.set_ylabel(r'$V_{th}$' + ' [mV]', color = 'r')
	ax4a.set_yticks(yticks)
	ax4a.tick_params(axis = 'y', labelcolor = 'r')

	ax5 = fig.add_subplot(spec[2, 1])

	plt.plot(network.E_rec.t/second, network.E_rec.Vm[attractor[1]]/mV, lw = 1.5, c = 'b')

	ax5.set_ylabel(r'$V_{mem}$' + ' [mV]', color = 'b')
	ax5.set_xlabel('time [s]')
	ax5.set_yticks(yticks)
	ax5.tick_params(axis = 'y', labelcolor = 'b')

	ax5a = ax5.twinx()

	plt.plot(network.E_rec.t/second, network.E_rec.Vth_e[attractor[1]]/mV, lw = 1.5, c = 'r')

	ax5a.set_ylabel(r'$V_{th}$' + ' [mV]', color = 'r')
	ax5a.set_yticks(yticks)
	ax5a.tick_params(axis = 'y', labelcolor = 'r')

	ax6 = fig.add_subplot(spec[0, 1])

	plt.plot(network.E_rec.t/second, network.E_rec.Vm[attractor[2]]/mV, lw = 1.5, c = 'b')

	ax6.set_ylabel(r'$V_{mem}$' + ' [mV]', color = 'b')
	ax6.set_xlabel('time [s]')
	ax6.set_yticks(yticks)
	ax6.tick_params(axis = 'y', labelcolor = 'b')

	ax6a = ax6.twinx()

	plt.plot(network.E_rec.t/second, network.E_rec.Vth_e[attractor[2]]/mV, lw = 1.5, c = 'r')

	ax6a.set_ylabel(r'$V_{th}$' + ' [mV]', color = 'r')
	ax6a.set_yticks(yticks)
	ax6a.tick_params(axis = 'y', labelcolor = 'r')

	# --- 3rd column --------------------------------------------------------------------

	# ax7 = fig.add_subplot(spec[0, 2])

	# _ = 1/(1+np.exp(u_mean))

	# # _ = (1 * np.exp(-np.exp(3.4578085977515953 - 6.480427880222709 * u_mean)))

	# plt.plot(t_array, _, label = 'f(u)')
	# plt.plot(t_array, u_mean, label = 'u')

	# plt.legend(loc = 'best', framealpha = 0)

	plt.show()
	plt.close()






