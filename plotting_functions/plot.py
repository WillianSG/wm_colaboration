import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import math
import sys, os
from brian2 import second, mV

if sys.platform == 'linux':
    parent_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '../')))
    sys.path.append(os.path.join(parent_dir, 'helper_functions'))

    from other import has_spiked
else:
    from helper_functions.other import has_spiked

"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Cognitive Modelling

Function:
- Plot: Creates a Plot object that can be used to plot data from the brian2 simulator.
- confidence_interval: Calculates a confidence interval for data series.
- plot_membrane_potentials: Builds plot to display post-synaptic membrane potentials Vm.
- plot_utilisation_resources: Builds plot to display synaptic variables and weights.
- plot_utilisation_resources: Builds plot to display synaptic weights.

Script arguments:
-

Script output:
-

Comments:
- For examples on how to use Plot to build plots, see the `plot_*` functions in this file.

Input:
- window: time window in seconds as tuple e.g., window=(0,3)*second
- data_monitor: the brian2 Monitor holding the data to plot
- neuron_monitor: the brian2 Monitor for the neuronal population
- sample_size: the number of neurons to sample among those which have fired in window
- synapse: the brian2 Synapse addressed by data_monitor
- num_top_plots: the number of top subplots
- num_bottom_plots: the number of bottom subplots

Output:
-
"""


class Plot:
    def __init__( self, window, data_monitor, neuron_monitor, sample_size=10, synapse=None,
                  num_top_plots=10, num_bottom_plots=1 ):
        self.window = window
        self.sample_size = sample_size
        self.data_monitor = data_monitor
        self.synapse = synapse
        
        self.current_top_subplot = -1
        self.current_bottom_subplot = -1
        
        self.spiked_idx = np.where( has_spiked( window, neuron_monitor ) )[ 0 ]
        self.neuron_sample = np.random.choice( self.spiked_idx, size=self.sample_size, replace=False )
        
        self.timepoints = \
            np.where( (self.data_monitor.t > self.window[ 0 ]) & (self.data_monitor.t < self.window[ 1 ]) )[ 0 ]
        self.t = self.data_monitor.t[ self.timepoints ]
        
        self.lwdth = 3
        self.s1 = 20
        self.s2 = 30
        mpl.rcParams[ 'axes.linewidth' ] = self.lwdth
        
        self.fig = plt.figure( figsize=(32, 22), constrained_layout=True )
        self.num_bottom_plots = num_bottom_plots
        self.num_top_plots = num_top_plots
        self.ncols = 5
        self.num_top_plots_lines = int( self.num_top_plots / self.ncols )
        self.nrows = self.num_bottom_plots + self.num_top_plots_lines
        self.gs = mpl.gridspec.GridSpec( nrows=self.nrows, ncols=self.ncols,
                                         hspace=0.1,
                                         # wspace=0.2,
                                         height_ratios=[ *([ 1 ] * (self.nrows - self.num_bottom_plots)),
                                                         *([ 1.5 ] * self.num_bottom_plots) ],
                                         figure=self.fig )
        self.gs.update( top=0.965, bottom=0.03 )
    
    def set_suptitle( self, suptitle ):
        self.fig.suptitle(
                f'{suptitle} t=[{self.window[ 0 ] * second}-{self.window[ 1 ] * second}] s',
                size=self.s2 )
    
    def add_top_plot( self, data, xlabel='', ylabel='', title='', colour='black' ):
        self.current_top_subplot += 1
        i, j = np.unravel_index( self.current_top_subplot, (self.num_top_plots, self.ncols) )
        
        ax = self.fig.add_subplot( self.gs[ i, j ] )
        ax.plot( self.t / second, data, color=colour )
        ax.set_xlabel( f'{xlabel}', size=self.s1, labelpad=5, horizontalalignment='center' )
        ax.set_ylabel( f'{ylabel}', size=self.s1, labelpad=5, horizontalalignment='center' )
        ax.set_title( f'{title}', fontsize=self.s1 )
        ax.tick_params( axis='both', which='major', labelsize=self.s1, width=self.lwdth, length=10, pad=15 )
    
    def add_bottom_mean_plot( self, data, xlabel='', ylabel='', title='', colour='black' ):
        self.current_bottom_subplot += 1
        i, j = np.unravel_index( self.current_bottom_subplot, (self.num_bottom_plots, 1) )
        
        ci = confidence_interval( data )
        ax = self.fig.add_subplot( self.gs[ self.num_top_plots_lines + i, : ] )
        ax.plot( self.t / second, np.mean( data, axis=0 ), linewidth=1, color=colour, label='Mean' )
        ax.fill_between( self.t / second, ci[ 0 ], ci[ 1 ], facecolor=colour, alpha=0.5, label='95% CI' )
        ax.set_xlabel( f'{xlabel}', size=self.s1, labelpad=5, horizontalalignment='center' )
        ax.set_ylabel( f'{ylabel}', size=self.s1, labelpad=5, horizontalalignment='center' )
        ax.set_title( title, fontsize=self.s1 )
        ax.tick_params( axis='both', which='major', labelsize=self.s1, width=self.lwdth, length=10, pad=15 )
        ax.legend()
    
    def get_variable( self, variable, filter_by='neuron', sampled=True, time_sliced=True, index_by_neuron=False ):
        first_filter = self.neuron_sample if sampled else self.spiked_idx
        if not index_by_neuron:
            second_filter = first_filter if filter_by == 'neuron' else self.synapse[ first_filter, : ]
            filtered_monitor = self.data_monitor[ second_filter ]
            filtered_variable = getattr( filtered_monitor, variable )
            if time_sliced:
                try:
                    return filtered_variable[ :, self.timepoints ]
                except:
                    return filtered_variable[ self.timepoints ]
        else:
            neurons_to_vars = { }
            for i in first_filter:
                second_filter = i if filter_by == 'neuron' else self.synapse[ i, : ]
                filtered_monitor = self.data_monitor[ second_filter ]
                neurons_to_vars[ i ] = getattr( filtered_monitor, variable )
                if time_sliced:
                    try:
                        neurons_to_vars[ i ] = neurons_to_vars[ i ][ :, self.timepoints ]
                    except:
                        neurons_to_vars[ i ] = neurons_to_vars[ i ][ self.timepoints ]
            
            return neurons_to_vars
    
    def set_title( self ):
        pass
    
    def save_plot( self, path, name ):
        self.fig.savefig(
                path + f'/{name}_{self.window[ 0 ] * second}-{self.window[ 1 ] * second}-s.png',
                bbox_inches='tight' )
    
    def show_plot( self ):
        self.fig.show()


def confidence_interval( data, cl=0.95 ):
    degrees_freedom = len( data ) - 1
    sampleMean = np.mean( data, axis=0 )
    sampleStandardError = st.sem( data, axis=0 )
    eps = np.finfo( sampleStandardError.dtype ).eps
    sampleStandardError[ sampleStandardError == 0 ] = eps
    confidenceInterval = st.t.interval( alpha=cl, df=degrees_freedom, loc=sampleMean, scale=sampleStandardError )
    
    return confidenceInterval


# TODO we are only interested in the synapses WITHIN the attractor, not the ones going out
def plot_membrane_potentials( window, data_monitor, neuron_monitor, show_plot=True, save_path='' ):
    plot = Plot( window, data_monitor, neuron_monitor, sample_size=10, num_bottom_plots=1 )
    plot.set_suptitle( 'Membrane potentials' )
    
    Vms = plot.get_variable( 'Vm', index_by_neuron=True )
    
    for i, v in Vms.items():
        plot.add_top_plot( v, ylabel='Vm', colour='magenta', title=f'Neuron {i}' )
    
    Vms = plot.get_variable( 'Vm', sampled=False )
    plot.add_bottom_mean_plot( Vms, xlabel='Time (s)', ylabel='Mean membrane conductance', title='All neurons',
                               colour='magenta' )
    
    if show_plot:
        plot.show_plot()
    
    if save_path:
        plot.save_plot( save_path, 'rcn_Vepsp' )


def plot_utilisation_resources( window, synapse, data_monitor, neuron_monitor, show_plot=True, save_path='' ):
    plot = Plot( window, data_monitor, neuron_monitor, synapse=synapse, sample_size=5, num_top_plots=10,
                 num_bottom_plots=4 )
    plot.set_suptitle( 'Synaptic variables' )
    
    us = plot.get_variable( 'u', filter_by='synapse', index_by_neuron=True )
    xs = plot.get_variable( 'x_', filter_by='synapse', index_by_neuron=True )
    
    for i, u in us.items():
        plot.add_top_plot( np.mean( u, axis=0 ), ylabel='Utilisation (u)', colour='blue', title=f'Neuron {i}' )
    for i, x in xs.items():
        plot.add_top_plot( np.mean( x, axis=0 ), xlabel='Time (s)', ylabel='Resources (x)', colour='orange' )
    
    us = plot.get_variable( 'u', filter_by='synapse', sampled=False )
    xs = plot.get_variable( 'x_', filter_by='synapse', sampled=False )
    ws = plot.get_variable( 'w', filter_by='synapse', sampled=False )
    
    plot.add_bottom_mean_plot( us, ylabel='Mean utilisation', title='All neurons', colour='blue' )
    plot.add_bottom_mean_plot( xs, ylabel='Mean resources', colour='orange' )
    plot.add_bottom_mean_plot( ws, ylabel='Mean weight', colour='green' )
    plot.add_bottom_mean_plot( ws * us * xs, xlabel='Time (s)', ylabel='Mean total effect', colour='black' )
    
    if show_plot:
        plot.show_plot()
    
    if save_path:
        plot.save_plot( save_path, 'rcn_syn_vars' )


def plot_synaptic_weights( window, synapse, data_monitor, neuron_monitor, show_plot=True, save_path='' ):
    plot = Plot( window, data_monitor, neuron_monitor, synapse=synapse, sample_size=10, num_top_plots=10,
                 num_bottom_plots=1 )
    plot.set_suptitle( 'Synaptic weights' )
    
    ws = plot.get_variable( 'w', filter_by='synapse', index_by_neuron=True )
    
    for i, w in ws.items():
        print(w, len(w))
        plot.add_top_plot( np.mean( w, axis=0 ), xlabel='Time (s)', ylabel='Weight (w)', colour='green',
                           title=f'Neuron {i}' )
    
    ws = plot.get_variable( 'w', filter_by='synapse', sampled=False )
    
    plot.add_bottom_mean_plot( ws, xlabel='Time (s)', ylabel='Mean weight', title='All neurons', colour='green' )
    
    if show_plot:
        plot.show_plot()
    
    if save_path:
        plot.save_plot( save_path, 'rcn_syn_weights' )
