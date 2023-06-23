import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms
import pickle
import pandas as pd
import matplotlib.colors as colors

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../')))

sys.path.append(os.path.join(root, 'helper_functions'))

path = 'D://A_PhD//GitHub//wm_colaboration//results//sFSA_surface_gauss_search'

files = os.listdir(path)

dict_ = {}
params = []

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    _CR = 0

    for param, val in pickle_dat.items():

        if param not in dict_:
            dict_[param] = []

            if param != 'CR':
            	params.append(param)

        dict_[param].append(val)

used_pairs = []

for param_x in params:
	for param_y in params:

		if param_y != param_x and (param_y, param_x) not in used_pairs:

			used_pairs.append((param_x, param_y))

			opt_x = []
			opt_y = []

			for i in range(len(dict_[param_x])):

				if dict_['CR'][i] == 1.0:

					opt_x.append(dict_[param_x])
					opt_y.append(dict_[param_y])

			fig = plt.figure()
			ax = fig.add_subplot(111)

			norm = colors.TwoSlopeNorm(vmin = 0, vcenter = 0.5, vmax = 1.0)

			surf = ax.scatter(dict_[param_x], dict_[param_y], 
				c = dict_['CR'],
                norm = norm,
                cmap = 'viridis',
                marker = 's')

			ax.axvline(x = np.round(np.mean(opt_x), 2), color = 'k', lw = 1.0, ls = '--')
			ax.axhline(y = np.round(np.mean(opt_y), 2), color = 'k', lw = 1.0, ls = '--')

			fig.colorbar(surf, label = 'CR')

			ax.set_xlabel(param_x)
			ax.set_xlim(np.min(dict_[param_x]), np.max(dict_[param_x]))
			ax.set_ylabel(param_y)
			ax.set_ylim(np.min(dict_[param_y]), np.max(dict_[param_y]))

			fig.tight_layout()
			plt.show()