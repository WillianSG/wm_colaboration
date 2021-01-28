# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
@original: adapted from Lehfeldt

Comments:
- [1] Adaptive threshold of E as in https://brian2.readthedocs.io/en/2.0rc/examples/adaptive_threshold.html
- What's the "external attractor net"?
- https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators
- https://stackoverflow.com/questions/6392739/what-does-the-at-symbol-do-in-python
- [IMPORTANT] tag with stuff to fix
- [ALTERED] tag with stuff alterade due to apparent original mistake
- [ADDED] tag with stuff added due to apparent original mistake
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
import random
from time import localtime

prefs.codegen.target = 'numpy'

# Helper modules
from store_simulation_settings import *
from load_stimulus import *
from load_parameters import *
from load_synapse_model import *
from return_w_matrix import *
from return_rho_matrix import *
from return_xpre_matrix import *
from return_xpost_matrix import *

class AttractorNetwork:
	def __init__(self):
		# 1 ========== Simulation settings ==========

		self.path_sim_id = ''
		self.path_sim = ''
		self.id = ''

		self.ext_att_freq = 0*Hz # activation freq. of external attractor (?)

		# 1.1 ========== General settings

		self.exp_type = 'user-defined' # type of experiment
		self.t_run = 1*second # duration of simul.
		self.int_meth_neur = 'user-defined' # NeuronGroup numerical integration method
		self.int_meth_syn = 'user-defined' # E_E synapses numerical integration method
		self.dt = 0.1*ms # Delta t of clock intervals

		# 1.2 ========== Spike monitors

		self.Input_to_E_mon_record = False
		self.Input_to_I_mon_record = False
		self.E_mon_record = False
		self.I_mon_record = False
		self.Ext_att_record = False

		# 1.3 ========== State monitors

		self.Input_E_rec_record = False
		self.Input_E_rec_attributes = ('w')
		self.Input_I_rec_record = False
		self.Input_I_rec_attributes = ('w')
		self.E_rec_record = False
		self.E_rec_attributes = ('Vm','Vepsp','Vipsp')
		self.I_rec_record = False
		self.I_rec_attributes = ('Vm','Vepsp','Vipsp')
		self.E_E_rec_record = False
		self.E_E_rec_attributes = ('w')
		self.E_I_rec_record = False
		self.E_I_rec_attributes = ('w')
		self.I_E_rec_record = False
		self.I_E_rec_attributes = ('w')
		self.Ext_att_E_rec_record = False
		self.Ext_att_E_rec_attributes = ('w')
		self.rec_dt = 0.1*ms # Delta t of state variable recording

		# 1.4 ========== @network_operation settings

		# Rho snapshots
		self.rho_matrix_snapshots = False
		self.rho_matrix_snapshots_step = 100 *ms  

		# Weight snapshots
		self.w_matrix_snapshots = False        
		self.w_matrix_snapshots_step = 100*ms  

		# Xpre snapshots
		self.xpre_matrix_snapshots = False        
		self.xpre_matrix_snapshots_step = 100*ms  

		# Xpost snapshots
		self.xpost_matrix_snapshots = False        
		self.xpost_matrix_snapshots_step = 100*ms  

		# Switches of stimulation protocols for testing attractors
		# Stimulus pulse
		self.stimulus_pulse = False
		self.stimulus_pulse_duration = 1*second
		self.stimulus_pulse_clock_dt = 1*second

		# 1.5 ========== Excitatory (E) stimulus

		self.stim_type_e = 'user-defined' # [IMPORTANT] - origi. don't have it
		self.stim_size_e = 0
		self.stim_freq_e = 0*Hz
		self.stim_offset_e = 0

		# 1.6 ========== Inhibitory (I) stimulus

		self.stim_type_i = 'user-defined'
		self.stim_size_i = 0
		self.stim_freq_i = 0*Hz
		self.stim_offset_i = 0

		# 2 ========== Network Settings ==========

		# Optional external population (copy of E_E wit fixed attractor)
		"""
		If True, adds Ext_att population for testing of ETF function
		"""
		self.add_Ext_att = False

		# 2.1 ========== Learning rule

		self.plasticity_rule = 'user-defined' # 'new', 'none'
		self.parameter = 'user-defined'

		# 2.1.1 - Rule parameters
		"""
		loaded in init_learning_rule_parameters()
		"""
		self.tau_rho = None
		self.thr_b_rho = None
		self.xpre_jump = None
		self.xpost_jump = None
		self.tau_xpre = None 
		self.tau_xpost = None
		self.alpha = None
		self.beta = None
		self.xpre_factor = None
		self.thr_post = None
		self.thr_pre = None
		self.rho_init = None
		self.rho_neg = None
		self.rho_neg2 = None
		self.rho_min = 0
		self.rho_max = 1
		self.bistabilty = True

		# 2.2 ========== Neurons

		self.neuron_type = 'user-defined' # 'LIF', 'poisson', 'spikegenerators'
		self.N_input_e = 256  # num. of input neurons for E
		self.N_input_i = 64   # num. of input neurons for I

		# 2.2.1 - Excitatory population [1]

		self.N_e = 256 # num. of neurons
		self.Vr_e = -65*mV # resting potential
		self.Vrst_e = -65*mV # reset potential
		self.Vth_e_init = -52*mV # initial threshold voltage
		self.Vth_e_incr = 5*mV # post-spike threshold voltage increase
		self.tau_Vth_e = 20*ms # time constant of threshold decay
		self.taum_e = 20*ms # membrane time constant
		self.tref_e = 2*ms # refractory period
		self.tau_epsp_e = 3.5*ms # time constant of EPSP
		self.tau_ipsp_e = 5.5*ms # time constant of IPSP

		# 2.2.2 - Inhibitory population

		self.N_i = 64 # num. of neurons
		self.Vr_i = -60*mV # resting voltage
		self.Vrst_i = -60*mV # reset potential
		self.Vth_i = -40*mV # threshold voltage
		self.taum_i = 10*ms # membrane time constant
		self.tref_i = 1*ms # refractory period
		self.tau_epsp_i = 3.5*ms # time constant of EPSP
		self.tau_ipsp_i = 5.5*ms # time constant of IPSP

		# 2.3 ========== Synapses

		self.syn_delay_Vepsp_e_e = 0*ms 
		self.syn_delay_w_update_e_e = 0*ms

		self.w_e_e_max = 1*mV # upper weight boundary for E_E connections
		self.w_ext_att_e = 0*mV # weight external attractor to attractor in E

		# Connection probabilities
		self.p_e_e = 0.4 # excitatory to excitatory
		self.p_e_i = 0.25 # excitatory to inhibitory
		self.p_i_e = 0.25 # inhibitory to excitatory

		# Connection weights
		self.w_input_e = 0*mV # input to excitatory
		self.w_input_i = 0*mV # input to inhibitory
		self.w_e_e = 0*mV # excitatory to excitatory
		self.w_e_i = 0*mV # excitatory to inhibitory
		self.w_i_e = 0*mV # inhibitory to excitatory

	# ========== Network Initialization ==========
	def init_network_modules(self):
		self.set_neurons()
		self.set_synapses()
		self.set_learning_rule_parameters()
		self.set_stimulus_e()
		self.set_stimulus_i()

		if self.add_Ext_att:
			self.set_stimulus_ext_att()

		self.set_weights()
		self.set_monitors()
		self.set_namespace()
		self.set_network_object()

	# ========== Neurons Initialization ==========
	def set_neurons(self):
		# 1 ========== Model equations ==========
		# Excitatory: LIF model with adaptive threshold
		self.eqs_e = Equations('''
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / taum_e : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt
			dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt''',
			Vr_e = self.Vr_e,
			taum_e = self.taum_e,
			tau_epsp = self.tau_epsp_e,
			tau_ipsp = self.tau_ipsp_e,
			Vth_e_init = self.Vth_e_init,
			tau_Vth_e = self.tau_Vth_e)

		# Inhibitory: LIF model  
		self.eqs_i = Equations(''' 
			dVm/dt = (Vepsp - Vipsp - (Vm - Vr_i)) / taum_i : volt (unless refractory)
			dVepsp/dt = -Vepsp / tau_epsp : volt
			dVipsp/dt = -Vipsp / tau_ipsp : volt''',
			Vr_i = self.Vr_i,
			taum_i = self.taum_i,
			tau_epsp = self.tau_epsp_i,
			tau_ipsp = self.tau_ipsp_i)

		# 2 ========== Neuronal populations ==========
		# Excitatory input population: poisson neurons
		self.Input_to_E = NeuronGroup(N = self.N_input_e,
			model = 'rates : Hz',
			threshold = 'rand()<rates*dt', 
			name = 'Input_to_E')

		# Inhibitory input population: poisson neurons
		self.Input_to_I = NeuronGroup(N = self.N_input_i, 
			model = 'rates : Hz', 
			threshold = 'rand()<rates*dt', 
			name = 'Input_to_I')

		# External attractor network (conditional)
		if self.add_Ext_att:
			self.N_ext_att = self.N_e 
			self.Ext_att = NeuronGroup(N = self.N_ext_att, 
				model = 'rates : Hz', 
				threshold = 'rand()<rates*dt', 
				name = 'Ext_att')

		# Excitatory population
		self.E = NeuronGroup(N = self.N_e, model = self.eqs_e,
			reset = '''Vm = Vrst_e 
					Vth_e += Vth_e_incr''',
			threshold = 'Vm > Vth_e',
			refractory = self.tref_e,
			method = self.int_meth_neur, 
			name = 'E')

		# Inhibitory population
		self.I = NeuronGroup(N = self.N_i, model = self.eqs_i,
			reset = 'Vm = Vrst_i',
			threshold = 'Vm > Vth_i',
			refractory = self.tref_i,
			method = self.int_meth_neur, 
			name = 'I')

		# 3 ========== Neuronal Attributes ==========
		# Adaptive threshold of E
		self.E.Vth_e = self.Vth_e_init

		# Random membrane voltages
		self.E.Vm = (self.Vrst_e + rand(self.N_e) * (self.Vth_e_init - self.Vr_e)) 
		self.I.Vm = (self.Vrst_i + rand(self.N_i) * (self.Vth_i - self.Vr_i))

		# Postsynaptic potentials
		self.E.Vepsp = 0
		self.E.Vipsp = 0
		self.I.Vepsp = 0
		self.I.Vipsp = 0

	# ========== Synapses Initialization ==========
	def set_synapses(self):
		# 1 ========== Loading model equations ==========
		# Load E_E model equations
		[self.model_E_E, self.pre_E_E, self.post_E_E] = load_synapse_model(
			plasticity_rule = self.plasticity_rule,
			neuron_type = self.neuron_type,
			bistability = self.bistability)

		# 2 ========== Creating synapses ==========
		# Non-plastic synapses
		self.Input_E = Synapses(source = self.Input_to_E, target = self.E, 
			model = 'w : volt', 
			on_pre = 'Vepsp += w', 
			name = 'Input_E')

		self.Input_I = Synapses(source = self.Input_to_I, target = self.I, 
			model = 'w : volt', 
			on_pre = 'Vepsp += w', 
			name = 'Input_I')

		self.E_I = Synapses(source = self.E, target = self.I, 
			model = 'w : volt', 
			on_pre = 'Vepsp += w', 
			name = 'E_I')

		self.I_E = Synapses(source = self.I, target = self.E, 
			model = 'w : volt', 
			on_pre = 'Vipsp += w', 
			name = 'I_E')

		if self.add_Ext_att:
			self.Ext_att_E = Synapses(source = self.Ext_att, target = self.E, 
				model = 'w : volt', 
				on_pre = 'Vepsp += w', 
				name = 'Ext_att_E')

		# Plastic synapses
		self.E_E = Synapses(
			source = self.E, 
			target = self.E, 
			model = self.model_E_E, 
			on_pre = self.pre_E_E,
			on_post = self.post_E_E,
			method = self.int_meth_syn,
			name = 'E_E')

		# 3 ========== Creating connections ==========
		self.Input_E.connect(j = 'i')
		self.Input_I.connect(j = 'i')
		self.E_E.connect('i!=j', p = self.p_e_e) # no self-connections
		self.E_I.connect(True, p = self.p_e_i)
		self.I_E.connect(True, p = self.p_i_e)

		if self.add_Ext_att:
			# assign E to E synaptic connections
			self.Ext_att_E.connect(i = self.E_E.i[:], j = self.E_E.j[:])

		self.E_E.Vepsp_transmission.delay = self.syn_delay_Vepsp_e_e

		if self.add_Ext_att:
			self.Ext_att_E.Vepsp_transmission.delay = self.syn_delay_Vepsp_e_e

		# Initialising calcium and efficacy variables
		if self.plasticity_rule == 'LR1' or self.plasticity_rule == 'LR2':
			self.E_E.xpre = 0
			self.E_E.xpost = 0
			self.E_E.rho = 0

	# ========== Rule's parameters loader ==========
	def set_learning_rule_parameters(self):
		if self.plasticity_rule == 'LR1' or  self.plasticity_rule =='LR2':
			self.num_E_E_synapses = len(self.E_E.i)

			[self.tau_xpre,
			self.tau_xpost,
			self.xpre_jump,
			self.xpost_jump,
			self.rho_neg,
			self.rho_neg2,
			self.rho_init,
			self.tau_rho,
			self.thr_post,
			self.thr_pre,
			self.thr_b_rho,
			self.rho_min,
			self.rho_max,
			self.alpha,
			self.beta, 
			self.xpre_factor,
			self.w_max] = load_rule_params(self.plasticity_rule, 
				self.parameter)

	# ========== Excitatory stimulus loader ==========
	def set_stimulus_e(self):
		# Reset input pattern to zero
		self.Input_to_E.rates = 0*Hz

		# Load stimulus to E 
		self.stim_inds_original_E = load_stimulus(
			stimulus_type = self.stim_type_e,
			stimulus_size = self.stim_size_e,
			offset = self.stim_offset_e)

		self.Input_to_E.rates[self.stim_inds_original_E] = self.stim_freq_e

	# ========== Inhibitory stimulus loader ==========
	def set_stimulus_i(self):
		# Load stimulus to I  
		self.stim_inds_original_I = load_stimulus(
			stimulus_type = self.stim_type_i,
			stimulus_size = self.stim_size_i,
			offset = self.stim_offset_i) 

		# Apply stimulus frequency to input neurons forming the stimulus  
		self.Input_to_I.rates[self.stim_inds_original_I] = self.stim_freq_i

	# ========== External attractor stimulus loader ==========
	def set_stimulus_ext_att(self):
		self.Ext_att.rates[self.stim_inds_original_E] = self.ext_att_freq

	# ========== Weights initializer ==========
	def set_weights(self):
		# Assign synaptic weights
		self.Input_E.w = self.w_input_e
		self.Input_I.w = self.w_input_i
		self.E_E.w = self.w_e_e
		self.E_I.w = self.w_e_i
		self.I_E.w = self.w_i_e

		if self.fixed_attractor:
			self.attractor_inds = load_stimulus(
				stimulus_type = self.stim_type_e,
				stimulus_size = self.stim_size_e,
				offset = self.stim_offset_e)

			self.att_src_inds = []
			self.att_tar_inds = []
			att_src_inds_collect = []

			# loop through attractor subnet
			for a in self.attractor_inds:
				# index positions of a
				src_inds_temp = np.where(self.E_E.i == a)
				# target neuron indices of a
				tar_inds_temp = self.E_E.j[src_inds_temp[0]]

				# Find positions of target neurons that belong to attractor
				att_tar_inds_collect = []

				# loop again through attractor subnet
				for b in self.attractor_inds:
					# position of target neuron belonging to attractor subnet
					att_tar_inds_temp=np.where(tar_inds_temp == b)

					# if target neuron is present, collect index position
					if len(att_tar_inds_temp[0])>0:
						# posit. within target neuron inds belonging to attrac.
						att_tar_inds_collect.append(att_tar_inds_temp[0][0])

				# Collect source positions belonging to the target positions
				for m in att_tar_inds_collect:
					src_temp=src_inds_temp[0][m]
					att_src_inds_collect.append(src_temp)

				# Collect all source positions
				self.att_tar_inds.extend(att_tar_inds_collect)
				self.att_src_inds.extend(att_src_inds_collect)

				if self.fixed_attractor_wmax == 'fraction_max': 
					self.w_assign = np.ones(len(self.att_src_inds)) * self.w_e_e

					self.w_high = np.random.choice(
						range(len(self.att_src_inds)),
						size = int(round(len(self.att_src_inds))* \
						self.fixed_attractor_w_frac),
						replace = False)

					self.w_assign[self.w_high] = self.w_e_e_max 
					self.E_E.w[self.att_src_inds] = self.w_assign

				if self.fixed_attractor_wmax == 'all_max':  
					self.E_E.w[self.att_src_inds] = self.w_e_e_max

		if self.add_Ext_att:
			self.Ext_att_E.w[self.att_src_inds] = self.w_ext_att_e[self.att_src_inds]

	# ========== Monitors initializer ==========
	def set_monitors(self):
		# Spikemonitors
		self.Input_to_E_mon = SpikeMonitor(source = self.Input_to_E,
			record = self.Input_to_E_mon_record, name = 'Input_to_E_mon')

		self.Input_to_I_mon = SpikeMonitor(source = self.Input_to_I,
			record = self.Input_to_I_mon_record, name = 'Input_to_I_mon')

		self.E_mon = SpikeMonitor(source = self.E, record = self.E_mon_record,
			name = 'E_mon')

		self.I_mon = SpikeMonitor(source = self.I, record = self.I_mon_record,name = 'I_mon')

		if self.add_Ext_att:
			self.Ext_att_mon = SpikeMonitor(source = self.Ext_att,
				record = self.Ext_att_record, name = 'Ext_att_mon')

			self.Ext_att_E_rec = StateMonitor(source = self.Ext_att_E,
				variables = self.Ext_att_E_rec_attributes,
				record = self.Ext_att_E_rec_record,
				dt = self.rec_dt,
				name = 'Ext_att_E_rec')

		# Statemonitors
		self.Input_E_rec = StateMonitor(source = self.Input_E,
			variables = self.Input_E_rec_attributes,
			record = self.Input_E_rec_record,
			dt = self.rec_dt,
			name = 'Input_E_rec')

		self.Input_I_rec = StateMonitor(source = self.Input_I,
			variables = self.Input_I_rec_attributes,
			record = self.Input_I_rec_record,
			dt = self.rec_dt,
			name = 'Input_I_rec')

		self.E_rec = StateMonitor(source = self.E,
			variables = self.E_rec_attributes,
			record = self.E_rec_record,
			dt = self.rec_dt,
			name = 'E_rec')

		self.I_rec = StateMonitor(source = self.I,
			variables = self.I_rec_attributes,
			record = self.I_rec_record,
			dt = self.rec_dt,
			name = 'I_rec')

		self.E_E_rec = StateMonitor(source = self.E_E,
			variables = self.E_E_rec_attributes,
			record = self.E_E_rec_record,
			dt = self.rec_dt,
			name = 'E_E_rec')

		self.E_I_rec = StateMonitor(source = self.E_I,
			variables = self.E_I_rec_attributes,
			record = self.E_I_rec_record,
			dt = self.rec_dt,
			name = 'E_I_rec')

		self.I_E_rec = StateMonitor(source = self.I_E,
			variables = self.I_E_rec_attributes,
			record = self.I_E_rec_record,
			dt = self.rec_dt,
			name = 'I_E_rec')

	# ========== Network namespace ==========
	def set_namespace(self):
		self.namespace = {
			'Vrst_e' : self.Vrst_e,
			'Vth_e_init' : self.Vth_e_init,
			'Vrst_i' : self.Vrst_i,
			'Vth_i' : self.Vth_i,
			'Vth_e_incr' : self.Vth_e_incr,
			'w_max' : self.w_e_e_max,
			'xpre_jump' : self.xpre_jump,
			'xpost_jump' : self.xpost_jump,
			'tau_xpre' : self.tau_xpre,
			'tau_xpost' : self.tau_xpost,
			'tau_rho' : self.tau_rho,
			'rho_min' : self.rho_min,
			'rho_max' : self.rho_max,
			'alpha' : self.alpha,
			'beta' : self.beta,
			'xpre_factor' : self.xpre_factor,
			'thr_b_rho' : self.thr_b_rho,
			'rho_neg' : self.rho_neg,
			'rho_neg2' : self.rho_neg2,
			'thr_post' : self.thr_post,
			'thr_pre' : self.thr_pre}

		return self.namespace

	# ========== @network_operation functions ==========
	def set_network_object(self):
		# ========== Clocks, counters and switches
		# Rho snapthots
		self.rho_matrix_snapshot_clock = Clock(
			self.rho_matrix_snapshots_step, name = 'clk_rho_snap')

		self.rho_matrix_snapshot_count = 0

		# Weight snapthots
		self.w_matrix_snapshot_clock = Clock(
			self.w_matrix_snapshots_step, name = 'clk_w_snap')

		self.w_matrix_snapshot_count = 0

		# Xpre snapthots
		self.xpre_matrix_snapshot_clock = Clock(
			self.xpre_matrix_snapshots_step, name = 'clk_xpre_snap')

		self.xpre_matrix_snapshot_count = 0

		# Xpost snapthots
		self.xpost_matrix_snapshot_clock = Clock(
			self.xpost_matrix_snapshots_step, name = 'clk_xpost_snap')

		self.xpost_matrix_snapshot_count = 0

		# Stimulus pulse
		self.stimulus_pulse_clock = Clock(
			self.stimulus_pulse_clock_dt, name = 'clk_stim_pulse')

		self.stimulus_pulse_switch = 0

		# ========== Snapshot of rho matrices
		if self.rho_matrix_snapshots:
			@network_operation(clock = self.rho_matrix_snapshot_clock)
			def store_rho_matrix_snapshot():
				data_rho_ee = return_rho_matrix('E_E', self.net,
					self.N_e, self.N_e)

				out_path = os.path.join(self.path_rho_snapshots,
					self.id + "_rho_snaps_E_E_" + \
					'{:02}'.format(self.rho_matrix_snapshot_count) + \
					"_time_" + '{:05.2f}'.format(defaultclock.t[:]/second) + \
					"s.csv")

				with open(out_path, 'wb') as f:
					np.savetxt(f, data_rho_ee, fmt = '%.3e', delimiter = ",")

				self.rho_matrix_snapshot_count += 1
		else:
			@network_operation(clock = self.rho_matrix_snapshot_clock)
			def store_rho_matrix_snapshot():
				pass

		# ========== Snapshot of weight matrices
		if self.w_matrix_snapshots:
			@network_operation(clock = self.w_matrix_snapshot_clock)
			def store_w_matrix_snapshot():
				data_w_ee = return_w_matrix('E_E', self.net, self.N_e, 
					self.N_e)

				out_path = os.path.join(self.path_w_snapshots,
					self.id + "_w_snaps_E_E_" + \
					'{:02}'.format(self.w_matrix_snapshot_count) + \
					"_time_" + '{:05.2f}'.format(defaultclock.t[:]/second) + \
					"s.csv")

				with open(out_path, 'wb') as f:
					np.savetxt(f, data_w_ee, fmt = '%.3e', delimiter = ",")

				self.w_matrix_snapshot_count += 1
		else:
			@network_operation(clock = self.w_matrix_snapshot_clock)
			def store_w_matrix_snapshot():
				pass

		# ========== Snapshot of xpre matrices
		if self.xpre_matrix_snapshots:
			@network_operation(clock = self.xpre_matrix_snapshot_clock)
			def store_xpre_matrix_snapshot():
				data_xpre_ee = return_xpre_matrix('E_E', self.net, self.N_e,
					self.N_e)

				out_path = os.path.join(self.path_xpre_snapshots, self.id + \
					"_xpre_snaps_E_E_" + \
					'{:02}'.format(self.xpre_matrix_snapshot_count) + \
					"_time_" + '{:05.2f}'.format(defaultclock.t[:]/second) + \
					"s.csv")

				with open(out_path, 'wb') as f:
					np.savetxt(f, data_xpre_ee, fmt = '%.3e', delimiter = ",")

				self.xpre_matrix_snapshot_count += 1
		else:
			@network_operation(clock = self.xpre_matrix_snapshot_clock)
			def store_xpre_matrix_snapshot():
				pass

		# ========== Snapshot of xpost matrices
		if self.xpost_matrix_snapshots:
			@network_operation(clock = self.xpost_matrix_snapshot_clock)
			def store_xpost_matrix_snapshot():
				data_xpost_ee = return_xpost_matrix('E_E', self.net, self.N_e,
					self.N_e)

				out_path = os.path.join(self.path_xpost_snapshots,self.id + \
					"_xpost_snaps_E_E_" + \
					'{:02}'.format(self.xpost_matrix_snapshot_count) + \
					"_time_" + '{:05.2f}'.format(defaultclock.t[:]/second) + \
					"s.csv")

				with open(out_path, 'wb') as f:
					np.savetxt(f, data_xpost_ee, fmt = '%.3e', delimiter = ",")

				self.xpost_matrix_snapshot_count += 1
		else:
			@network_operation(clock = self.xpost_matrix_snapshot_clock)
			def store_xpost_matrix_snapshot():
				pass

		# ========== Protocol 1 - one stimulus pulse, constant stimulus offset
		if self.stimulus_pulse:
			@network_operation(clock = self.stimulus_pulse_clock)
			def stimulus_pulse():
				if defaultclock.t >= self.stimulus_pulse_duration:
					self.stim_freq_e = 0*Hz
					self.set_stimulus_e()
		else:
			@network_operation(clock = self.stimulus_pulse_clock)
			def stimulus_pulse():
				pass

		# ========== Network object
		defaultclock.dt = self.dt

		if self.add_Ext_att:
			self.net = Network(
				self.Input_to_E,
				self.Input_to_I,
				self.E,
				self.I,
				self.Ext_att,
				self.Input_E,
				self.Input_I,
				self.E_E,
				self.E_I,
				self.I_E,
				self.Ext_att_E,
				self.Input_to_E_mon,
				self.Input_to_I_mon,
				self.E_mon,
				self.I_mon,
				self.Ext_att_mon,
				self.Input_E_rec,
				self.Input_I_rec,
				self.E_rec,
				self.I_rec,
				self.E_E_rec,
				self.E_I_rec,
				self.I_E_rec,
				self.Ext_att_E_rec,
				store_rho_matrix_snapshot,
				store_w_matrix_snapshot,
				store_xpre_matrix_snapshot,
				store_xpost_matrix_snapshot,
				stimulus_pulse,
				name = 'net')
		else:
			self.net = Network(
				self.Input_to_E,
				self.Input_to_I,
				self.E,
				self.I,
				self.Input_E,
				self.Input_I,
				self.E_E,
				self.E_I,
				self.I_E,
				self.Input_to_E_mon,
				self.Input_to_I_mon,
				self.E_mon,
				self.I_mon,
				self.Input_E_rec,
				self.Input_I_rec,
				self.E_rec,
				self.I_rec,
				self.E_E_rec,
				self.E_I_rec,
				self.I_E_rec,
				store_rho_matrix_snapshot,
				store_w_matrix_snapshot,
				store_xpre_matrix_snapshot,
				store_xpost_matrix_snapshot,
				stimulus_pulse,
				name = 'net')

	# ========== Results storage ==========
	def set_simulation_folder(self):
		self.iter_count = 0  # simulation iteration added to .txt name
		self.parent_dir = os.path.dirname(os.getcwd()) # parent directory
		self.simulation_folder = os.path.join(self.parent_dir, 'sim_network')

		if not(os.path.isdir(self.simulation_folder)):
			os.mkdir(self.simulation_folder)

		self.cwd = os.getcwd()  # current working directory
		self.idt = localtime()  # local date and time

		if self.id == '':
			self.id = str(self.idt.tm_year) + \
			'{:02}'.format(self.idt.tm_mon) + \
			'{:02}'.format(self.idt.tm_mday)+ '_' + \
			'{:02}'.format(self.idt.tm_hour)+ \
			'_' + '{:02}'.format(self.idt.tm_min) + '_' + \
			'{:02}'.format(self.idt.tm_sec)

		self.path_sim_id = os.path.join(self.simulation_folder, self.id)

		if not(os.path.isdir(self.path_sim_id)):
			os.mkdir(self.path_sim_id)

		# Rho, Ca and w snapshot's directory
		self.path_snapshots = os.path.join(self.path_sim_id, 
			self.id + "_snapshots")

		if not(os.path.isdir(self.path_snapshots)):
			os.mkdir(self.path_snapshots)

		# Directory for rho_matrix_snapshots 
		self.path_rho_snapshots = os.path.join(self.path_snapshots, 
			self.id + "_rho_snapshots")

		if not(os.path.isdir(self.path_rho_snapshots)):
			os.mkdir(self.path_rho_snapshots)

		# Directory for w_matrix_snapshots 
		self.path_w_snapshots = os.path.join(self.path_snapshots, 
			self.id + "_w_matrix_snapshots")

		if not(os.path.isdir(self.path_w_snapshots)):
			os.mkdir(self.path_w_snapshots)

		# Directory for xpre_matrix_snapshots 
		self.path_xpre_snapshots = os.path.join(self.path_snapshots, 
			self.id + "_xpre_snapshots")

		if not(os.path.isdir(self.path_xpre_snapshots)):
			os.mkdir(self.path_xpre_snapshots)

		# Directory for xpost_matrix_snapshots 
		self.path_xpost_snapshots = os.path.join(self.path_snapshots, 
			self.id + "_xpost_snapshots")

		if not(os.path.isdir(self.path_xpost_snapshots)):
			os.mkdir(self.path_xpost_snapshots)

	# ========== Run simulation ==========
	def run_net(self, report = 'stdout', period = 1):
		# ========== Storage location
		if os.path.isdir(self.path_sim_id):
			pass
		elif os.path.isdir(self.path_sim):
			pass
		else:
			self.set_simulation_folder()

		# Change path for storage of simulation settings as .txt-file
		if self.iter_count == 0:
			self.abs_path_sim_id = self.path_sim_id
		else:
			self.abs_path_sim_id = self.path_sim_id + '_' + self.exp_type

		# Initialisation of namespace before storage to have updated version
		self.set_namespace()

		store_simulation_settings(
			sim_id = os.path.join(self.abs_path_sim_id, self.id),
			exp_type = self.exp_type,
			iter_count = self.iter_count,
			t_run = self.t_run,
			int_meth_neur = self.int_meth_neur,
			int_meth_syn = self.int_meth_syn,
			dt = self.dt,
			Input_to_E_mon_record = self.Input_to_E_mon_record,
			Input_to_I_mon_record = self.Input_to_I_mon_record,
			E_mon_record = self.E_mon_record,
			I_mon_record = self.I_mon_record,
			Ext_att_record = self.Ext_att_record,
			Input_E_rec_record = self.Input_E_rec_record,
			Input_E_rec_attributes_orig = self.Input_E_rec_attributes,
			Input_I_rec_record = self.Input_I_rec_record,
			Input_I_rec_attributes_orig = self.Input_I_rec_attributes,
			E_rec_record = self.E_rec_record,
			E_rec_attributes_orig = self.E_rec_attributes,
			I_rec_record = self.I_rec_record,
			I_rec_attributes_orig = self.I_rec_attributes,
			E_E_rec_record = self.E_E_rec_record,
			E_E_rec_attributes_orig = self.E_E_rec_attributes,
			E_I_rec_record = self.E_I_rec_record,
			E_I_rec_attributes_orig = self.E_I_rec_attributes,
			I_E_rec_record = self.I_E_rec_record,
			I_E_rec_attributes_orig = self.I_E_rec_attributes,
			Ext_att_E_rec_record = self.Ext_att_E_rec_record,
			Ext_att_E_rec_attributes_orig = self.Ext_att_E_rec_attributes,
			rec_dt = self.rec_dt,
			rho_matrix_snapshots = self.rho_matrix_snapshots,
			rho_matrix_snapshots_step = self.rho_matrix_snapshots_step,
			w_matrix_snapshots = self.w_matrix_snapshots,
			w_matrix_snapshots_step = self.w_matrix_snapshots_step,
			xpre_matrix_snapshots = self.xpre_matrix_snapshots,
			xpre_matrix_snapshots_step = self.xpre_matrix_snapshots_step,
			xpost_matrix_snapshots = self.xpost_matrix_snapshots,
			xpost_matrix_snapshots_step = self.xpost_matrix_snapshots_step,
			stimulus_pulse = self.stimulus_pulse,
			stimulus_pulse_duration = self.stimulus_pulse_duration,
			stimulus_pulse_clock_dt = self.stimulus_pulse_clock_dt,
			stim_type_e = self.stim_type_e,
			stim_size_e = self.stim_size_e,
			stim_freq_e = self.stim_freq_e,
			stim_type_i = self.stim_type_i,
			stim_size_i = self.stim_size_i,
			stim_freq_i = self.stim_freq_i,
			ext_att_freq = self.ext_att_freq,
			plasticity_rule = self.plasticity_rule,
			neuron_type = self.neuron_type,
			N_input_e = self.N_input_e,
			N_input_i = self.N_input_i,
			N_e = self.N_e,
			Vr_e = self.Vr_e,
			Vrst_e = self.Vrst_e,
			Vth_e_init = self.Vth_e_init,
			tau_Vth_e = self.tau_Vth_e,
			Vth_e_incr = self.Vth_e_incr,
			taum_e = self.taum_e,
			tref_e = self.tref_e,
			tau_epsp_e = self.tau_epsp_e,
			tau_ipsp_e = self.tau_ipsp_e,
			N_i = self.N_i,
			Vr_i = self.Vr_i,
			Vrst_i = self.Vrst_i,
			Vth_i = self.Vth_i,
			taum_i = self.taum_i,
			tref_i = self.tref_i,
			tau_epsp_i = self.tau_epsp_i,
			tau_ipsp_i = self.tau_ipsp_i,
			p_e_e = self.p_e_e,
			p_e_i = self.p_e_i,
			p_i_e = self.p_i_e,
			w_input_e = self.w_input_e,
			w_input_i = self.w_input_i,
			w_e_e = self.w_e_e,
			w_e_i = self.w_e_i,
			w_i_e = self.w_i_e,
			w_e_e_max = self.w_e_e_max,
			w_ext_att_e = self.w_ext_att_e,
			add_Ext_att = self.add_Ext_att,
			tau_xpre = self.tau_xpre,
			tau_xpost = self.tau_xpost,
			tau_rho = self.tau_rho,
			thr_b_rho = self.thr_b_rho,
			xpre_jump = self.xpre_jump,
			xpost_jump = self.xpost_jump,
			xpre_factor   = self.xpre_factor,
			rho_neg = self.rho_neg,
			eqs_e = self.eqs_e,
			eqs_i = self.eqs_i,
			model_E_E = self.model_E_E,
			pre_E_E = self.pre_E_E,
			post_E_E = self.post_E_E,
			namespace = self.namespace,
			net = self.net)

		# go to w_matrix_snapshot folder (?)
		if self.rho_matrix_snapshots or self.w_matrix_snapshots or self.xpre_matrix_snapshots or self.xpost_matrix_snapshots:
			pass

		# running simulation
		self.net.run(
			self.t_run, 
			report = report, 
			report_period = period*second, 
			namespace = self.set_namespace())

		self.net.stop()

		# (?)
		if self.iter_count == 0:
			self.abs_path = self.path_sim_id
		else:
			self.abs_path = self.path_sim

		# ========== Delete snap folders if w and/or xpre/xpost not stored
		if self.rho_matrix_snapshots == False and self.iter_count == 0:
			self.abs_path_rho_snapshots = os.path.join(self.path_snapshots,
				self.path_rho_snapshots) # [ADDED]

			shutil.rmtree(self.path_rho_snapshots)

			if self.iter_count == 0:
				self.abs_path = self.path_sim_id
				# [ALTERED] - abs_path = self.path_sim_id
			else:
				self.abs_path = self.path_sim
				# [ALTERED] - abs_path = self.path_sim

		if self.w_matrix_snapshots == False and self.iter_count == 0:
			self.abs_path_w_snapshots = os.path.join(self.path_snapshots, 
				self.path_w_snapshots)

			shutil.rmtree(self.abs_path_w_snapshots)

			if self.iter_count == 0:
				self.abs_path = self.path_sim_id
			else:
				self.abs_path = self.path_sim

		if self.xpre_matrix_snapshots == False and self.iter_count == 0:
			self.abs_path_xpre_snapshots = os.path.join(self.path_snapshots,
				self.path_xpre_snapshots)

			shutil.rmtree(self.abs_path_xpre_snapshots)

			if self.iter_count == 0:
				self.abs_path = self.path_sim_id
			else:
				self.abs_path = self.path_sim

		if self.xpost_matrix_snapshots == False and self.iter_count==0: 
			abs_path_xpost_snapshots = os.path.join(self.path_snapshots, 
				self.path_xpost_snapshots)

			shutil.rmtree(abs_path_xpost_snapshots)

			if self.iter_count == 0:
				self.abs_path = self.path_sim_id
			else:
				self.abs_path = self.path_sim

		# Delete snapshot folder if neither rho nor w nor c snapshots were taken
		if self.rho_matrix_snapshots == False and self.w_matrix_snapshots == False and self.xpre_matrix_snapshots == False and self.xpost_matrix_snapshots == False and self.iter_count == 0:
			self.abs_path_snapshots = os.path.join(self.path_sim_id, 
				self.path_snapshots)

			shutil.rmtree(self.abs_path_snapshots)

		# Rename simulation folder: add information about type of experiment
		if self.iter_count == 0:
			self.path_sim = self.path_sim_id + '_' + self.exp_type
			self.abs_path_sim_id = os.path.join(self.simulation_folder, 
				self.path_sim_id)
			self.abs_path_sim = os.path.join(self.simulation_folder, 
				self.path_sim)

			os.rename(self.abs_path_sim_id, self.path_sim) 
		else:
			pass

		# Update iter_count
		self.iter_count += 1

		# Go back to working directory (?) {y commented out?}
		#os.chdir(self.cwd)