# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

import os
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../../')))

_simf_path = os.path.join(root, 'results/RCN_controlled_PS_grid_search_4')

_gs_sim_folders = os.listdir(_simf_path)

_ba_iew_dict = {}

for _simf in _gs_sim_folders:

	_simf_files = os.listdir(os.path.join(_simf_path, _simf))
	_folder = _simf_files[0]
	_simf_files = os.listdir(os.path.join(_simf_path, _simf, _folder))

	for _f in _simf_files:

		if _f.find('simulation_summary.txt') != -1:

			_txt = open(os.path.join(_simf_path, _simf, _folder, _f), 'r')
			_lines = _txt.readlines()

			for _l in _lines:

				if _l.find('background activity (Hz):') != -1:
					
					_ba = float(_l.split(': ')[-1])

				elif _l.find('cueing time (s):') != -1:
					
					_ct = float(_l.split(': ')[-1])

				elif _l.find('inh. to exc. weight (mV):') != -1:
					
					_ie_w = float(_l.split(': ')[-1])

				elif _l.find('attractor A1: ') != -1:
					
					_A1 = _l.split('attractor A1: ')[-1].replace('\n', '')
					_A1 = _A1.split(',')

					_tgr_A1 = int(_A1[0].split(': ')[-1])
					_spt_A1 = int(_A1[-1].split(': ')[-1].replace('}', ''))

				elif _l.find('attractor A2: ') != -1:
					
					_A2 = _l.split('attractor A2: ')[-1].replace('\n', '')
					_A2 = _A2.split(',')

					_tgr_A2 = int(_A2[0].split(': ')[-1])
					_spt_A2 = int(_A2[-1].split(': ')[-1].replace('}', ''))

			if _ct not in _ba_iew_dict:

				_ba_iew_dict[_ct] = {'A1': {'trg': [], 'spt': []}, 'A2': {'trg': [], 'spt': []}}

				_ba_iew_dict[_ct]['A1']['trg'].append(_tgr_A1)
				_ba_iew_dict[_ct]['A1']['spt'].append(_spt_A1)

				_ba_iew_dict[_ct]['A2']['trg'].append(_tgr_A2)
				_ba_iew_dict[_ct]['A2']['spt'].append(_spt_A2)

			else:

				_ba_iew_dict[_ct]['A1']['trg'].append(_tgr_A1)
				_ba_iew_dict[_ct]['A1']['spt'].append(_spt_A1)

				_ba_iew_dict[_ct]['A2']['trg'].append(_tgr_A2)
				_ba_iew_dict[_ct]['A2']['spt'].append(_spt_A2)

for key, val in _ba_iew_dict.items():

	_ba_iew_dict[key]['A1']['trg'] = {
		'mean': np.round(np.mean(_ba_iew_dict[key]['A1']['trg']), 2), 
		'std': np.round(np.std(_ba_iew_dict[key]['A1']['trg']), 2)
		}
	_ba_iew_dict[key]['A1']['spt'] = {
		'mean': np.round(np.mean(_ba_iew_dict[key]['A1']['spt']), 2), 
		'std': np.round(np.std(_ba_iew_dict[key]['A1']['spt']), 2)
		}

	_ba_iew_dict[key]['A2']['trg'] = {
		'mean': np.round(np.mean(_ba_iew_dict[key]['A2']['trg']), 2), 
		'std': np.round(np.std(_ba_iew_dict[key]['A2']['trg']), 2)
		}
	_ba_iew_dict[key]['A2']['spt'] = {
		'mean': np.round(np.mean(_ba_iew_dict[key]['A2']['spt']), 2), 
		'std': np.round(np.std(_ba_iew_dict[key]['A2']['spt']), 2)
		}

# for key, val in _ba_iew_dict.items():

# 	print(key, val['A2']['trg'])

# _ = np.arange(0.1, 1.1, 0.1)

# x_ = []
# y_ = []
# err_ = []

# for i in _:

# 	i = np.round(i, 1)

# 	x_.append(i)
# 	y_.append(_ba_iew_dict[i]['A1']['spt']['mean'])
# 	err_.append(_ba_iew_dict[i]['A1']['spt']['std'])

# x_A2 = []
# y_A2 = []
# err_A2 = []

# for i in _:

# 	i = np.round(i, 1)

# 	x_A2.append(i)
# 	y_A2.append(_ba_iew_dict[i]['A2']['trg']['mean'])
# 	err_A2.append(_ba_iew_dict[i]['A2']['trg']['std'])

# fig, ax = plt.subplots()

# ax.errorbar(x_, y_, yerr=err_)

# plt.xlabel('cueing time [s]')
# plt.ylabel('A1 spt spike [a.u.]')

# ax.set_xticks(_)
# ax.set_yticks(np.arange(0, 7, 1))

# ax2 = ax.twinx() 

# ax2.errorbar(x_A2, y_A2, yerr=err_A2, color = 'r')

# plt.show()



fig = plt.figure(constrained_layout = True, figsize = (10, 4))

spec2 = gridspec.GridSpec(
	ncols = 2, 
	nrows = 1, 
	width_ratios = [4, 4],
	height_ratios = [4],
	figure = fig)

ax1 = fig.add_subplot(spec2[0, 0], projection = '3d')

plt.title('attractor A1 \n trg reactivation \n average of 10 simlations')

x = []
y = []
z = []

c = []

for key, val in _ba_iew_dict.items():

	x.append(key[0])
	y.append(key[1])
	z.append(val['A2']['trg']['mean'])

	c.append(val['A2']['trg']['std'])

sc = ax1.scatter(x, y, z, c = c, marker = '^')

ax1.set_xlabel('background activity [Hz]')
ax1.set_ylabel('I to E w [mV]')
ax1.set_zlabel('trg PS count [a.u]')

fig.colorbar(sc, label = 'SD')

ax2 = fig.add_subplot(spec2[0, 1], projection = '3d')

plt.title('attractor A1 \n spont reactivation \n average of 10 simlations')

x = []
y = []
z = []

c = []

for key, val in _ba_iew_dict.items():

	x.append(key[0])
	y.append(key[1])
	z.append(val['A1']['spt']['mean'])

	c.append(val['A1']['spt']['std'])

sc = ax2.scatter(x, y, z, c = c, marker = 'v')

ax2.set_xlabel('background activity [Hz]')
ax2.set_ylabel('I to E w [mV]')
ax2.set_zlabel('spt PS count [a.u]')

fig.colorbar(sc, label = 'SD')

plt.show()
plt.close()
