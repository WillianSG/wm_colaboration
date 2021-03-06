# -*- coding: utf-8 -*-
"""
@author: wgirao

Inputs:

Outputs:

Comments:
"""

from brian2 import ms, mV, second
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))


sim_data = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\20210306_13_28_06_synaptic_matrix_after_flat_to_E_fixed_size_offset0_sec1.pickle'

with open(sim_data,'rb') as f:(
	synaptic_matrix) = pickle.load(f)

plt.imshow(synaptic_matrix, cmap='coolwarm')
plt.colorbar()
plt.savefig('C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\after_cue.png')
plt.show()

sim_data = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\20210306_13_28_06_synaptic_matrix_after_flat_to_E_fixed_size_offset42_sec3.pickle'

with open(sim_data,'rb') as f:(
	synaptic_matrix) = pickle.load(f)

plt.imshow(synaptic_matrix, cmap='coolwarm')
plt.colorbar()
plt.savefig('C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\after_two.png')
plt.show()

sim_data = 'C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\20210306_13_28_06_synaptic_matrix_after_flat_to_E_fixed_size_offset84_sec4.pickle'

with open(sim_data,'rb') as f:(
	synaptic_matrix) = pickle.load(f)

plt.imshow(synaptic_matrix, cmap='coolwarm')
plt.colorbar()
plt.savefig('C:\\Users\\willi\\PhD_Stuff\\learning_rule\\network_results\\attractor_stability\\20210306_13_28_06_attractor_stability\\after_all.png')
plt.show()

