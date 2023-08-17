# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

import pickle, sys
import numpy as np

try:
    seed = int(sys.argv[1])
except:
    seed = 0

np.random.seed(seed)

# - Generate FSA classifying digits.
fsa = {
    'S': ['A', 'B'],
    'I': ['0', '1'],
    'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']
}

init_state = 'A'

# - Function parsing bin. number to extract correct state transitions.
def compute_transition(current_state, input_symbol, fsa):

    match = f'({current_state}, {input_symbol})'

    for transition in fsa['T']:

        if transition.find(match) != -1:
            return transition.split('->')[-1]
        
# - Generating 6 bit binary numbers.
bits_number = 6
total_numbers = 100
all_digits = []

digits = np.random.randint(2, size = (bits_number*total_numbers))

for i in range(0, len(digits), bits_number):
    all_digits.append([str(x) for x in digits[i:i+bits_number]])

# - Parsing binary numbers.
all_state_history = []

for bin_number in all_digits:

    states_history = []

    states_history.append(init_state)

    for sym in bin_number:

        states_history.append(compute_transition(states_history[-1], sym, fsa))

    all_state_history.append(states_history)

# - Exporting digits and respective state transition hitory for defined FSA.

with open('even_odd_binary_numbers_FSA.pickle', 'wb') as f:
    pickle.dump({'digits': all_digits, 'states_history': all_state_history}, f)