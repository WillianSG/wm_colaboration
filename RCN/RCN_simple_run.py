import argparse

from helper_functions.recurrent_competitive_network import run_rcn

parser = argparse.ArgumentParser()
parser.add_argument('-background_activity', type=float, default=15.095621488966966,
                    help='Level of background activity in Hz')
parser.add_argument('-i_e_weight', type=float, default=5.011403421477593,
                    help='Weight of I-to-E synapses in mV')
parser.add_argument('-e_i_weight', type=float, default=4.9317965951712415,
                    help='Weight of E-to-I synapses in mV')
parser.add_argument('-e_e_max_weight', type=float, default=14.531800821256894,
                    help='Maximum weight of E-to-E synapses in mV')
parser.add_argument('-i_frequency', type=float, default=10.003928854474928,
                    help='Frequency of I input in Hz')
parser.add_argument('-cue_percentage', type=float, default=100,
                    help='Percentage of neurons in the attractor to be stimulated')
parser.add_argument('-cue_length', type=float, default=1,
                    help='Duration of the cue in seconds')
parser.add_argument('-num_attractors', type=int, default=3,
                    help='Number of attractors to be generated')
parser.add_argument('-attractor_size', type=int, default=64,
                    help='Number of neurons in each attractor')
parser.add_argument('-network_size', type=int, default=256,
                    help='Number of neurons in the network')
parser.add_argument('-num_cues', type=int, default=10,
                    help='Number of cues given to the network')
args = parser.parse_args()

params = {k: v for k, v in vars(args).items()}

rcn, returned_params = run_rcn(params, show_plot=True, progressbar=True, low_memory=False,
                               attractor_conflict_resolution="3", return_complete_stats=True)
