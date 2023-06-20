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

_reps = 50

bar = Bar(
	'state transition', 
	max = _reps*2)

for i in range(_reps):

	os.system('py RCN_state_transition.py')

	bar.next()

	os.system('py RCN_state_transition.py --cue_A1 0')
	
	bar.next()

bar.finish()