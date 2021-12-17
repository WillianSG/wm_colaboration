# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- Script sets a plastic synapse operating under the rule _plasticity_rule_ between a pair of pre_neurons and
postsynaptic neurons spiking with Poisson (normally distributed) spike trains, with firing frequencies of _pre_rate_
(Hz) and _post_rate_ (Hz), respectively.
- The code runs <num_sim> independent simulations of 1 second each. The state variables associated with the weight
and calcium traces are saved tracked during each simulation and averaged out to build plots meant to show the
behavior of the aforementioned variables during neuron activity.

Script arguments:
- pre_rate      [int(sys.argv[1])]: firing freq. (Hz) of presynaptic neuron.
- post_rate     [int(sys.argv[2])]: firing freq. (Hz) of postsynaptic neuron.
- num_sim       [int(sys.argv[3])]: number of independent simulation runs.
- rule          [str(sys.argv[4])]: learning rule.
- parameter_set [str(sys.argv[5])]: learning rule parameter set.

Script output:
- The output of the script will appear in a folder called **synapse_results** one level up where the script resides (
learning_rule/synapse_results). Grey curves represent data from each of the independent simulations, whereas colored
thick curves represent the average of all <num_sim> simulations.
"""
import setuptools
import os, sys, pickle
import platform
from brian2 import *
from scipy import *
from numpy import *
from joblib import Parallel, delayed
from time import localtime, strftime
import multiprocessing

prefs.codegen.target = 'numpy'

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

pre_rate = int( sys.argv[ 1 ] )
post_rate = int( sys.argv[ 2 ] )
num_sim = int( sys.argv[ 3 ] )
rule = str( sys.argv[ 4 ] )
parameter_set = str( sys.argv[ 5 ] )

# 0 ========== User degined imports ==========

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname( os.getcwd() )

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append( os.path.join( parent_dir, helper_dir ) )

# interesting_graph_results dir check
results_path = os.path.join( parent_dir, 'synapse_results' )

is_dir = os.path.isdir( results_path )
if not (is_dir):
	os.mkdir( results_path )

# Creating simulation ID
idt = localtime()
sim_id = strftime( "%d%b%Y_%H-%spikemon_P-%S_", localtime() )

# Starts a new scope for magic functions
start_scope()

# Helper modules
from load_rule_parameters import *
from load_synapse_model import *
from load_neurons import *

# 1 ========== Execution parameters ==========

exp_type = 'rates1'  # 'showcase', 'rates1'
dt_resolution = 0.001  # secs | simulation step time resolution
t_run = 1  # simulation time (seconds)
int_meth_syn = 'euler'  # Synaptic integration method

N_Pre = 1  # presynaptic neurons
N_Post = 1  # postsynaptic neurons

# 1.1 ========== Rule's parameters

plasticity_rule = rule  # 'none', 'LR1', 'LR2', 'LR3'
bistability = True
stoplearning = True

[ tau_xpre,
  tau_xpost,
  xpre_jump,
  xpost_jump,
  rho_neg,
  rho_neg2,
  rho_init,
  tau_rho,
  thr_post,
  thr_pre,
  thr_b_rho,
  rho_min,
  rho_max,
  alpha,
  beta,
  xpre_factor,
  w_max,
  xpre_min,
  xpost_min,
  xpost_max,
  xpre_max,
  tau_xstop,
  xstop_jump,
  xstop_max,
  xstop_min,
  thr_stop_h,
  thr_stop_l ] = load_rule_parameters(
		plasticity_rule=plasticity_rule,
		parameter_set=parameter_set )

w_init = w_max * rho_init

# 1.2 ========== net parameters

if exp_type == 'showcase':
	neuron_type = 'spikegenerator'
else:
	neuron_type = 'poisson'

# 2 ========== Learning rule as Brian2's synaptic model ==========
[ model_E_E,
  pre_E_E,
  post_E_E ] = load_synapse_model( plasticity_rule, neuron_type, bistability, stoplearning=stoplearning )

# 3 ========== Brian2's neuron objects

# only used when 'exp_type' is set to 'showcase'
input_pre = np.array( [ 10, 15, 18, 25, 27, 29, 31, 32 ] ) / 1000
input_post = np.array( [ 15, 95, 105, 250 ] ) / 1000

Pre, Post = load_neurons(
		N_Pre, N_Post, neuron_type,
		spikes_t_Pre=input_pre,
		spikes_t_Post=input_post,
		pre_rate=pre_rate,
		post_rate=post_rate )

# 3.1 ========== setting connections between neurons
Pre_Post = Synapses(
		source=Pre,
		target=Post,
		model=model_E_E,
		on_pre=pre_E_E,
		on_post=post_E_E,
		method=int_meth_syn,
		name='Pre_Post' )

Pre_Post.connect( j='i' )

Pre_Post.plastic = True
Pre_Post.rho = rho_init
Pre_Post.w = w_init

# 3.2 ========== Setting simulation monitors

if stoplearning:
	StateMon = StateMonitor( Pre_Post, [ 'xpre', 'xstop', 'xpost', 'w', 'rho' ], record=True )
else:
	StateMon = StateMonitor( Pre_Post, [ 'xpre', 'xpost', 'w', 'rho' ], record=True )

Pre_spk_mon = SpikeMonitor(
		source=Pre,
		record=True,
		name='Pre_spk_mon' )

Post_spk_mon = SpikeMonitor(
		source=Post,
		record=True,
		name='Post_spk_mon' )

# 4. ========== Running network ==========

store()

rho_all = [ ]
xpost_all = [ ]
xpre_all = [ ]
xstop_all = [ ]

for x in range( 0, num_sim ):
	print( '> running sim #', x )
	restore()
	
	run( t_run * second )
	
	rho_all.append( StateMon.rho[ 0 ] )
	xpost_all.append( StateMon.xpost[ 0 ] )
	xpre_all.append( StateMon.xpre[ 0 ] )
	
	if stoplearning:
		xstop_all.append( StateMon.xstop[ 0 ] )

# ================== avg rho

n_pot = 0
n_dep = 0

avg_pot_mag = 0  # avg potentiation magnitude
avg_dep_mag = 0  # avg depression magnitude

rho_all = np.array( rho_all )

avg_rho = np.zeros( len( rho_all[ 0 ] ) )

for row in rho_all:
	avg_rho += row
	if row[ -1 ] > row[ 0 ]:
		n_pot += 1
		avg_pot_mag += row[ -1 ] - row[ 0 ]
	elif row[ -1 ] < row[ 0 ]:
		n_dep += 1
		avg_dep_mag += row[ 0 ] - row[ -1 ]
	else:
		pass

if n_dep > 0:
	avg_dep_mag = avg_dep_mag / n_dep
else:
	avg_dep_mag = 0.0

if n_pot > 0:
	avg_pot_mag = avg_pot_mag / n_pot
else:
	avg_pot_mag = 0.0

if n_pot == 0:
	pot_perc = 0.0
else:
	pot_perc = int( (n_pot / num_sim) * 100 )

if n_dep == 0:
	dep_perc = 0.0
else:
	dep_perc = int( (n_dep / num_sim) * 100 )

avg_rho = avg_rho / num_sim

pot_avg_perc_change = np.round( (avg_pot_mag * 100) / rho_all[ 0 ][ 0 ], 2 )
dep_avg_perc_change = np.round( (avg_dep_mag * 100) / rho_all[ 0 ][ 0 ], 2 )

xpre_all = np.array( xpre_all )
xpost_all = np.array( xpost_all )

avg_xpre = np.zeros( len( xpre_all[ 0 ] ) )
avg_xpost = np.zeros( len( xpost_all[ 0 ] ) )

if stoplearning:
	avg_xstop = np.zeros( len( xstop_all[ 0 ] ) )

for x in range( 0, num_sim ):
	avg_xpre += xpre_all[ x ]
	avg_xpost += xpost_all[ x ]
	
	if stoplearning:
		avg_xstop += xstop_all[ x ]

avg_xpre = avg_xpre / num_sim
avg_xpost = avg_xpost / num_sim

if stoplearning:
	avg_xstop = avg_xstop / num_sim

# 5. ========== Plots ==========

# ============================ statistics ================================

fig0 = plt.figure( constrained_layout=True )

widths = [ 8, 8 ]
heights = [ 8, 8 ]

spec2 = gridspec.GridSpec(
		ncols=2,
		nrows=2,
		width_ratios=widths,
		height_ratios=heights,
		figure=fig0 )

if stoplearning:
	widths = [ 8, 8 ]
	heights = [ 8, 8, 8 ]
	
	spec2 = gridspec.GridSpec(
			ncols=2,
			nrows=3,
			width_ratios=widths,
			height_ratios=heights,
			figure=fig0 )

# fig0.suptitle('Param. set ' + parameter_set, fontsize = 8)

# avg rho
f2_ax1 = fig0.add_subplot( spec2[ 0, 0 ] )

for row in rho_all:
	plt.plot( StateMon.t, row, color='lightgrey', linestyle='--', linewidth=0.5 )

plt.plot( StateMon.t, avg_rho, color='k', linestyle='-', label='$\\rho_{avg}$' )

plt.hlines( rho_all[ 0 ][ 0 ], 0, StateMon.t[ -1 ], color='k', linestyle='--', label='$\\rho_{init}$' )

f2_ax1.legend( prop={ 'size': 5 } )

f2_ax1.set_ylim( [ 0.0, 1.0 ] )

plt.yticks( np.arange( 0.0, 1.2, step=0.2 ) )

plt.ylabel( 'rho (a.u.)', size=6 )
plt.xlabel( 'time (sec)', size=6 )
plt.title( '$\\rho$ evolution', size=8 )

# pot/dep %
f2_ax2 = fig0.add_subplot( spec2[ 0, 1 ] )

f2_ax2.pie( [ pot_perc, dep_perc ],
            labels=[ '+ (' + str( pot_avg_perc_change ) + '%)', '- (' + str( dep_avg_perc_change ) + '%)' ],
            autopct='%1.1f%%',
            shadow=True, startangle=90, colors=[ 'lightblue', 'tomato' ] )

f2_ax2.axis( 'equal' )

plt.title( 'Pot. vs Dep.', size=8 )

# avg Ca pre_neurons
f2_ax3 = fig0.add_subplot( spec2[ 1, 0 ] )

for row in xpre_all:
	plt.plot( StateMon.t, row, color='lightgrey', linestyle='--', linewidth=0.5 )

plt.plot( StateMon.t, avg_xpre, color='lightcoral', linestyle='-', label='$Ca^{2+}_{avg}$' )

plt.hlines( thr_pre, 0, StateMon.t[ -1 ], color='lightcoral', linestyle='--', label='$\\theta_{pre_neurons}$' )

plt.ylabel( '$Ca^{2+}_{pre_neurons}$', size=6 )
plt.xlabel( 'time (sec)', size=6 )
plt.title( '$Ca^{2+}_{pre_neurons}$ evolution (' + str( pre_rate ) + 'Hz)', size=8 )

f2_ax3.set_ylim( [ 0.0, 1.0 ] )

plt.yticks( np.arange( 0.0, 1.2, step=0.2 ) )

f2_ax3.legend( prop={ 'size': 5 } )

# avg Ca post
f2_ax4 = fig0.add_subplot( spec2[ 1, 1 ] )

for row in xpost_all:
	plt.plot( StateMon.t, row, color='lightgrey', linestyle='--', linewidth=0.5 )

plt.plot( StateMon.t, avg_xpost, color='lightblue', linestyle='-', label='$Ca^{2+}_{avg}$' )

plt.hlines( thr_post, 0, StateMon.t[ -1 ], color='lightblue', linestyle='--', label='$\\theta_{post}$' )

plt.ylabel( '$Ca^{2+}_{post}$', size=6 )
plt.xlabel( 'time (sec)', size=6 )
plt.title( '$Ca^{2+}_{post}$ evolution (' + str( post_rate ) + 'Hz)', size=8 )

f2_ax4.set_ylim( [ 0.0, 1.0 ] )

plt.yticks( np.arange( 0.0, 1.2, step=0.2 ) )

f2_ax4.legend( prop={ 'size': 5 } )

# Stop learning
if stoplearning:
	f2_ax5 = fig0.add_subplot( spec2[ 2, 1 ] )
	
	for row in xstop_all:
		plt.plot( StateMon.t, row, color='lightgrey', linestyle='--', linewidth=0.5 )
	
	plt.plot( StateMon.t, avg_xstop, color='green', linestyle='-', label='$Ca^{stop}_{avg}$' )
	
	plt.hlines( thr_stop_h, 0, StateMon.t[ -1 ], color='green', linestyle='-.', label='$\\theta_{stop}^{h}$' )
	
	plt.hlines( thr_stop_l, 0, StateMon.t[ -1 ], color='green', linestyle='--', label='$\\theta_{stop}^{l}$' )
	
	plt.ylabel( '$Ca^{stop}$', size=6 )
	plt.xlabel( 'time (sec)', size=6 )
	plt.title( '$Ca^{stop}$ evolution (' + str( post_rate ) + 'Hz)', size=8 )
	
	f2_ax5.set_ylim( [ 0.0, 1.0 ] )
	
	plt.yticks( np.arange( 0.0, 1.2, step=0.2 ) )
	
	f2_ax5.legend( prop={ 'size': 5 } )

fig0.suptitle(
		'rule: ' + str( rule ) + ' | parameters: ' + str( parameter_set ),
		fontsize=10 )

plot_name = sim_id + '_statistics_' + str( num_sim ) + '_' + plasticity_rule + '_' + parameter_set.replace( '.',
                                                                                                            '-' ) + \
            '_bist' + str(
		bistability ) + '_stopl' + str( stoplearning ) + '_pre' + str( pre_rate ) + '_post' + str( post_rate )

plt.savefig( os.path.join( results_path, plot_name ),
             bbox_inches='tight',
             dpi=200 )

print( '\n> script ended.' )
