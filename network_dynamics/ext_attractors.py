# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag

Comments:
- Adapted from Lehfeldt
- "Ext_attractors is to look at varying stimulus duration for a given maximum weight."
"""
import setuptools
import os, sys, pickle
from brian2 import *
from numpy import *
from time import localtime

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'

# Helper modules
from network import Network

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Starts a new scope for magic functions
start_scope()

# 1 ========== Execution parameters ==========

nets = 1 # number of networks ('?')
t_run = 10*second # simulation time
w_max = 7.25*mV # max weight in EE connections

# Stimulus pulse setttings (?)
pulse_duration_min = 3 # zero seconds not possible
pulse_duration_max = 3
pulse_duration_step = 1
pulse_durations = arange(pulse_duration_min, pulse_duration_max+pulse_duration_step, pulse_duration_step)*second

# 2 ========== Initialize and store network (?) ==========

for i in arange(0, nets, 1):
	print('Simulating network ', i, ' of ', nets)

	net = Network() # network class
