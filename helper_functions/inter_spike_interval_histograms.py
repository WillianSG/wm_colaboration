# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- isi: list containing inter spike intervals
- b: number of bins

Output:
- isi_hist_c: inter spike interval histogram as count
- edgs_c: bin edges of isi_hist_c
- b_wdth: bin width of edgs_c
- isi_hist_p: inter spike interval histogram as percentage

Comments:
- Calculates inter spike interval histograms from lists of inter spike intervals.
- Input population: Inter spike interval histogram .
"""
import numpy
import numpy as np

def inter_spike_interval_histograms(isi, b):
	isi_hist_c, edgs_c = numpy.histogram(isi, bins = b)

	b_wdth = edgs_c[1] - edgs_c[0]

	isi_hist_p = np.divide(isi_hist_c, float(np.sum(isi_hist_c)))

	return isi_hist_c, edgs_c, b_wdth, isi_hist_p
