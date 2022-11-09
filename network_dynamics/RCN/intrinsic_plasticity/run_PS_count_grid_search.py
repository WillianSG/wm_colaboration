# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""
import os, sys
from progress.bar import Bar

_ba = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
_iew = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]

reps = 10

bar = Bar(
	'stability dropped w (G, sync)', 
	max = len(_ba)*len(_iew)*reps)

for ba in _ba:

	for iew in _iew:

		for i in range(0, reps):

			os.system('python3 RCN_intrinsic_adaptation_simulation.py --ba_amount {} --i_amount {}'.format(ba, iew))
	
			bar.next()

bar.finish()