import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import second, ms
import pickle
import pandas as pd

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../')))

sys.path.append(os.path.join(root, 'helper_functions'))

path = 'D://A_PhD//GitHub//wm_colaboration//results//sFSA_gauss_search'

files = os.listdir(path)

post_proc_optimal = {}
post_proc_subopt = {}

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    _CR = 0

    for param, val in pickle_dat.items():

        if param == 'CR':

            _CR = val

            break

    for param, val in pickle_dat.items():

        if param != 'CR':

            if param not in post_proc_optimal:
                post_proc_optimal[param] = []

            if param not in post_proc_subopt:
                post_proc_subopt[param] = []

            if _CR == 1:

                post_proc_optimal[param].append(val)

            else:

                post_proc_subopt[param].append(val)

for key, val in post_proc_optimal.items():

    _subopt = post_proc_subopt[key]
    _opt = post_proc_optimal[key]

    _max = np.max(_subopt + _opt)
    _min = np.min(_subopt + _opt)

    plt.hist(_subopt, color = 'r', alpha = 0.35)
    plt.axvline(x = np.round(np.mean(_subopt), 2), color = 'r', lw = 1.0)
    plt.text(np.round(np.mean(_subopt), 2), 2, str(np.round(np.mean(_subopt), 2)), 
        fontsize = 15, color = 'r')

    plt.hist(_opt, color = 'b', alpha = 0.35)
    plt.axvline(x = np.round(np.mean(_opt), 2), color = 'b', lw = 1.0)
    plt.text(np.round(np.mean(_opt), 2), 5, str(np.round(np.mean(_opt), 2)), 
        fontsize = 15, color = 'b')

    plt.xlim(_min, _max)

    plt.xlabel(key)

    plt.tight_layout()
    plt.show()
    plt.close()

# sub_df = pd.DataFrame.from_dict(post_proc_subopt)
# opt_df = pd.DataFrame.from_dict(post_proc_optimal)

# opt_df.hist(color = 'b')
# for ax in plt.gcf().axes:
#    ax.set_title(ax.get_title().replace('Histogram of ', ''))
#    ax.set_xlabel('Values')
#    ax.set_ylabel('Frequency')
   

# plt.tight_layout()
# plt.show()
# plt.close()


# sub_df.hist(color = 'r')
# for ax in plt.gcf().axes:
#    ax.set_title(ax.get_title().replace('Histogram of ', ''))
#    ax.set_xlabel('Values')
#    ax.set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()
# plt.close()