# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- Just shows the dynamics of the syn variables x and u.

To Do:
- [] post as LIF neuron to test 'Vepsp_transmission_LR4 = {'Vepsp_transmission' : '''Vepsp += w*(u*x_)'''}'
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

pre_rate = 0
post_rate = 0
num_sim = 1
rule = 'LR4'
parameter_set = '1.0'

# 0 ========== User degined imports ==========

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname( os.getcwd() )

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append( os.path.join( parent_dir, helper_dir ) )

# Results dir check
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
from load_neurons_v2 import *

# 1 ========== Execution parameters ==========

neuron_type = 'LIF'
int_meth_syn = 'euler'  # Synaptic integration method

dt_resolution = 0.001  # secs | simulation step time resolution
t_run = 1  # simulation time (seconds)

N_Pre = 1  # presynaptic neurons
N_Post = 1  # postsynaptic neurons

# 1.1 ========== Rule's parameters

plasticity_rule = rule  # 'none', 'LR1', 'LR2', 'LR3'
bistability = True
stoplearning = False
resources = True

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
  thr_stop_l,
  U,
  tau_d,
  tau_f ] = load_rule_parameters(
        plasticity_rule=plasticity_rule,
        parameter_set=parameter_set )

rho_init = 1.0
w_max = 20 * mV
w_init = w_max * rho_init

# 2 ========== Learning rule as Brian2's synaptic model ==========
[ model_E_E,
  pre_E_E,
  post_E_E ] = load_synapse_model(
        plasticity_rule,
        neuron_type,
        bistability,
        stoplearning=stoplearning,
        resources=resources )

# 3 ========== Brian2's neuron objects

input_pre = np.array( [ 100, 140, 180, 220, 260, 300, 340, 540 ] ) / 1000
Pre, Post = load_neurons( spikes_t_Pre=input_pre )

Vr_e = -65 * mV  # resting potential
taum_e = 20 * ms  # membrane time constant
tau_epsp_e = 3.5 * ms  # time constant of EPSP
tau_ipsp_e = 5.5 * ms  # time constant of IPSP
Vth_e_init = -52 * mV  # initial threshold voltage
tau_Vth_e = 20 * ms  # time constant of threshold decay
Vrst_e = -65 * mV  # reset potential
Vth_e_incr = 5 * mV  # post-spike threshold voltage increase

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

Pre_Post.x_ = 1.0  # so that it starts with value 1.0
Pre_Post.u = U  # so that it starts with value U

Pre_Post.plastic = True
Pre_Post.rho = rho_init
Pre_Post.w = w_init

# 3.2 ========== Setting simulation monitors

StateMon = StateMonitor( Pre_Post, [ 'xpre', 'xpost', 'w', 'rho', 'x_', 'u' ], record=True )

Pre_spk_mon = SpikeMonitor(
        source=Pre,
        record=True,
        name='Pre_spk_mon' )

Post_spk_mon = SpikeMonitor(
        source=Post,
        record=True,
        name='Post_spk_mon' )

post_rec = StateMonitor(
        source=Post,
        variables=[ 'Vm' ],
        record=True,
        dt=0.1 * ms,
        name='post_rec' )

run( t_run * second )

fig = plt.figure( constrained_layout=True, figsize=(5, 8) )

widths = [ 8 ]
heights = [ 0.25, 2.5, 0.25, 1.5 ]

spec2 = gridspec.GridSpec(
        ncols=1,
        nrows=4,
        width_ratios=widths,
        height_ratios=heights,
        figure=fig )

ax1 = fig.add_subplot( spec2[ 0, 0 ] )

y = [ 0 ] * len( Pre_spk_mon.t )
plt.plot( Pre_spk_mon.t / ms, y, 'k|' )
plt.xlabel( 'Time (ms)' )
plt.ylabel( 'pre_neurons spikes' )
ax1.set_yticklabels( [ ] )

ax2 = fig.add_subplot( spec2[ 1, 0 ] )

plt.plot( StateMon.t / ms, StateMon.x_[ 0 ], label='x', color='tomato' )
plt.plot( StateMon.t / ms, StateMon.u[ 0 ], label='u', color='blue' )
plt.legend()
plt.xlabel( 'Time (ms)' )
plt.ylabel( 'synaptic variables' )

ax3 = fig.add_subplot( spec2[ 2, 0 ] )

y = [ 0 ] * len( Post_spk_mon.t )
plt.plot( Post_spk_mon.t / ms, y, 'k|' )
plt.xlabel( 'Time (ms)' )
plt.ylabel( 'post spikes' )

ax3.set_yticklabels( [ ] )

ax4 = fig.add_subplot( spec2[ 3, 0 ] )

plt.plot( post_rec.t / ms, post_rec.Vm[ 0 ] )
plt.xlabel( 'Time (ms)' )
plt.ylabel( r'$V_{mem}$ post (mV)' );

fig.suptitle(
        'rule: ' + str( rule ) + ' | parameters: ' + str( parameter_set ),
        fontsize=10 )

plt.show()
