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
- spikes_t_Pre(np.array): pres- spike times (default - single synapse timing experiment like)
- spikes_t_Post(np.array): post- spike times (default - single synapse timing experiment like)
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
from check_spikes_isi import *


def load_neurons(N_Pre, N_Post, neuron_type, spikes_t_Pre=[i / 1000.0 for i in [20, 120, 340, 540]],
                 spikes_t_Post=[i / 1000.0 for i in [40, 130, 320, 530]], pre_rate=1, post_rate=1):
    if neuron_type == 'spikegenerator':
        spikes_t_Post = check_spikes_isi(spikes_t_Post, 0.0001)
        spikes_t_Pre = check_spikes_isi(spikes_t_Pre, 0.0001)

        spikes_t_Pre = np.array(spikes_t_Pre.flatten()) * second
        spikes_t_Post = np.array(spikes_t_Post.flatten()) * second

        inds_PRE = np.zeros(len(spikes_t_Pre))
        inds_POST = np.zeros(len(spikes_t_Post))

        Pre = SpikeGeneratorGroup(
            N=N_Pre,
            indices=inds_PRE,
            times=spikes_t_Pre,
            name='Pre')

        Post = SpikeGeneratorGroup(
            N=N_Post,
            indices=inds_POST,
            times=spikes_t_Post,
            name='Post')

    elif neuron_type == 'poisson':
        Pre = NeuronGroup(
            N=N_Pre,
            model='rates1 : Hz',
            threshold='rand()<rates1*dt',
            name='Pre')

        Post = NeuronGroup(
            N=N_Post,
            model='rates1 : Hz',
            threshold='rand()<rates1*dt',
            name='Post')

        Pre.rates = pre_rate * Hz
        Post.rates = post_rate * Hz
    else:
        raise ValueError("\nneuron type should be 'poisson' or 'spikegenerator'")

    return Pre, Post
