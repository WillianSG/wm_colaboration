# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Creates both pre_neurons- and postsynaptic neuron groups (Brian2's 'NeuronGroup' method).
"""
input:
- N_Pre(int): # of neurons on presynaptic group
- N_Post(int): # of neurons on postynaptic group
- neuron_type(str): type of neuron ('poisson', 'LIF' , 'spikegenerators')
- spikes_t_Pre(np.array): pres- spike times (default - single synapse timing experiment like)
- spikes_t_Post(np.array): post- spike times (default - single synapse timing experiment like)
- pre_rate(int/default=1): pre_neurons- firing rate (in Hz?)
- post_rate(int/default=1): post- firing rate (in Hz?)

output:
- Pre: Brian2's neuron population object
- Post: Brian2's neuron population object

Comments:
"""
from brian2 import SpikeGeneratorGroup, NeuronGroup, Equations, rand
from brian2.units import second, ms, Hz, fundamentalunits, mV
import numpy as np
from check_spikes_isi import *


def load_neurons( spikes_t_Pre=[ i / 1000.0 for i in [ 20, 120, 340, 540 ] ] ):
    spikes_t_Pre = check_spikes_isi( spikes_t_Pre, 0.0001 )
    spikes_t_Pre = np.array( spikes_t_Pre.flatten() ) * second
    
    inds_PRE = np.zeros( len( spikes_t_Pre ) )
    
    Pre = SpikeGeneratorGroup(
            N=1,
            indices=inds_PRE,
            times=spikes_t_Pre,
            name='Pre' )
    
    Vr_e = -65 * mV  # resting potential
    taum_e = 20 * ms  # membrane time constant
    tau_epsp_e = 3.5 * ms  # time constant of EPSP
    tau_ipsp_e = 5.5 * ms  # time constant of IPSP
    Vth_e_init = -52 * mV  # initial threshold voltage
    tau_Vth_e = 20 * ms  # time constant of threshold decay
    Vrst_e = -65 * mV  # reset potential
    Vth_e_incr = 5 * mV  # post-spike threshold voltage increase
    
    eqs_e = Equations( '''
		dVm/dt = (Vepsp - Vipsp - (Vm - Vr_e)) / taum_e : volt (unless refractory)
		dVepsp/dt = -Vepsp / tau_epsp : volt
		dVipsp/dt = -Vipsp / tau_ipsp : volt
		dVth_e/dt = (Vth_e_init - Vth_e) / tau_Vth_e : volt''',
                       Vr_e=Vr_e,
                       taum_e=taum_e,
                       tau_epsp=tau_epsp_e,
                       tau_ipsp=tau_ipsp_e,
                       Vth_e_init=Vth_e_init,
                       tau_Vth_e=tau_Vth_e )
    
    Post = NeuronGroup(
            N=1,
            model=eqs_e,
            reset='''Vm = Vrst_e
				Vth_e += Vth_e_incr''',
            threshold='Vm > Vth_e',
            refractory=2 * ms,
            method='linear',
            name='Post' )
    
    Post.Vm = (Vrst_e + rand( 1 ) * (Vth_e_init - Vr_e))
    
    return Pre, Post
