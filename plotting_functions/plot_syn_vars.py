# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
-

Script arguments:
-

Script output:
-
"""

import os, pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_syn_vars(path, show=False):
    path_u = 'stimulus_neur_u.pickle'

    pickled_file = os.path.join(path, path_u)

    with open(pickled_file, 'rb') as f:
        (
            us,
            mon_t) = pickle.load(f)

    path_x = 'stimulus_neur_x.pickle'

    pickled_file = os.path.join(path, path_x)

    with open(pickled_file, 'rb') as f:
        (
            xs,
            mon_t) = pickle.load(f)

    for u in us:
        plt.plot(mon_t, u, "b")
    plt.plot([], [], "b", label="Calcium (u)")
    for x in xs:
        plt.plot(mon_t, x, "r")
    plt.plot([], [], "r", label="Resources (x)")

    plt.legend()

    if show:
        plt.show()

    plt.savefig(
        os.path.join(path, 'synaptic_variables.png'),
        bbox_inches='tight')

    plt.close()
