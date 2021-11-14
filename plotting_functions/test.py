# -*- coding: utf-8 -*-
from brian2 import second, ms
from plot_firing_rates_distribution import *

plot_firing_rates_distribution(
	target_data = '12Nov2021_19-06-00_RCN',
	t_run = 1*second,
	bin_width = 100*ms)