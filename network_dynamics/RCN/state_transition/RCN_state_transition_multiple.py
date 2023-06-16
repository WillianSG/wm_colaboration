# -*- coding: utf-8 -*-
"""
###########################################################################
#                                                                         #
# @author: w.soares.girao@rug.nl                                          #
# @university: University of Groningen                                    #
# @group: Bio-Inspired Circuits and Systems (BICS)                        #
#                                                                         #
###########################################################################
"""

import os
from progress.bar import Bar

_reps = 10

bar = Bar(
	'A1 -> A2 -> GO (1)', 
	max = _reps)

for i in range(_reps):

	os.system('py RCN_state_transition.py')
	
	bar.next()

bar.finish()

bar = Bar(
	'A1 -> A2 -> GO (2)', 
	max = _reps)

for i in range(_reps):

	os.system('py RCN_state_transition.py --cue_A1 0')
	
	bar.next()

bar.finish()