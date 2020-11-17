# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Creates both pre- and postsynaptic neuron groups (Brian2's 'NeuronGroup' method).
"""
input:
- N_Pre(int): # of neurons on presynaptic group
- N_Post(int): # of neurons on postynaptic group
- neuron_type(str): type of neuron ('poisson', 'LIF' , 'spikegenerators')
- spikes_t_Pre(np.array): pres- spike times
- spikes_t_Post(np.array): post- spike times
- pre_rate(int/default=1): pre- firing rate (in Hz?)
- post_rate(int/default=1): post- firing rate (in Hz?)

output:
- Pre: Brian2's neuron population object
- Post: Brian2's neuron population object

Comments:
"""
from brian2 import SpikeGeneratorGroup, NeuronGroup
from brian2.units import second, ms, Hz, fundamentalunits
import numpy as np

def load_neurons(N_Pre, N_Post, neuron_type, spikes_t_Pre = [i/1000.0 for i in [20, 120 ,340, 540]], spikes_t_Post = [i/1000.0 for i in [40, 130, 320, 530]], pre_rate = 1, post_rate = 1):

	print(spikes_t_Pre)

	# if neuron_type == 'spikegenerator':