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

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_conn_matrix(conn_matrix, population, path):
	plot = plt.imshow(
		conn_matrix , 
		cmap = 'bwr' , 
		interpolation = 'nearest')

	plt.title(population + ' connection matrix')

	plt.colorbar(plot)

	plt.clim(np.min(conn_matrix), np.max(conn_matrix))

	plt.savefig(
		os.path.join(path, population + '_conn_matrix.png'), 
		bbox_inches = 'tight')

	plt.close()
