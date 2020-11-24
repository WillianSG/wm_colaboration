# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""
import os
from numpy import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

# TO DO - have 'em as arguments for the script
# Target file
rho_file = "rho_LR1_2.1_6785564.npy"
drho_file = "drho_LR1_2.1_6785564.npy"

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Dir with .npy data with results of simulations
sim_results_dir = parent_dir + '\\sim_results\\'

# Results dir check
results_path = parent_dir + '\\plots'
is_dir = os.path.isdir(results_path)
if not(is_dir):
	os.mkdir(results_path)

# Loading data
final_rho_all = load(sim_results_dir + rho_file)
drho_all = load(sim_results_dir + drho_file)



print(final_rho_all)
print(drho_all)