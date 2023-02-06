# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""
import os, sys
from progress.bar import Bar

# _ba = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
# _iew = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]

# reps = 10

# bar = Bar(
# 	'Wie & BA', 
# 	max = len(_ba)*len(_iew)*reps)

# for ba in _ba:

# 	for iew in _iew:

# 		for i in range(0, reps):

# 			os.system('python3 RCN_intrinsic_adaptation_simulation.py --ba_amount {} --i_amount {} --i_stim_amount 20'.format(ba, iew))
	
# 			bar.next()

# bar.finish()

cue_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

reps = 10

bar = Bar(
	'cue time', 
	max = len(cue_times)*reps)

for ct in cue_times:

	for i in range(0, reps):

		os.system('python3 RCN_intrinsic_adaptation_simulation.py --ba_amount 15 --i_amount 10 --i_stim_amount 20 --A2_cue_time {}'.format(ct))

		bar.next()

bar.finish()