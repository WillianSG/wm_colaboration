import brian2.monitors
import numpy as np
import matplotlib.pyplot as plt
import os, datetime

"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Cognitive Modelling

Function:
- Has neuron spiked in the supplied time window?

Script arguments:
-

Script output:
- has_spiked: dictionary with neurons as keys and True or False depending if the neuron has spiked or not within the
time window

Comments:
- 

Inputs:
- window(tuple, list, Numpy array, or int, float): 2-element list-like object containing endpoints of time window,
if not list-like then the lower bound is assumed t=0
- data_monitor(brian2.data_monitor.SpikeMonitor): Brian2 SpikeMonitor object with recorded spikes
"""


def has_spiked( window, monitor ):
    assert isinstance( monitor, brian2.monitors.SpikeMonitor )
    assert isinstance( window, brian2.Quantity )
    assert window.size == 1 or window.size == 2
    if window.size == 1:
        window = (0, window)
    
    spikes = monitor.spike_trains()
    has_spiked = np.zeros( len( spikes ), dtype=bool )
    for i, spks in spikes.items():
        sp = spks[ (spks > window[ 0 ]) & (spks < window[ 1 ]) ]
        # print(i, np.count_nonzero(sp))
        has_spiked[ i ] = bool( np.count_nonzero( sp ) )
    
    return has_spiked


def visualise_connectivity( S ):
    Ns = len( S.source )
    Nt = len( S.target )
    plt.figure( figsize=(10, 4) )
    plt.subplot( 121 )
    plt.plot( np.zeros( Ns ), np.arange( Ns ), 'ok', ms=10 )
    plt.plot( np.ones( Nt ), np.arange( Nt ), 'ok', ms=10 )
    for i, j in zip( S.i, S.j ):
        plt.plot( [ 0, 1 ], [ i, j ], '-k' )
    plt.xticks( [ 0, 1 ], [ 'Source', 'Target' ] )
    plt.ylabel( 'Neuron index' )
    plt.xlim( -0.1, 1.1 )
    plt.ylim( -1, max( Ns, Nt ) )
    plt.subplot( 122 )
    plt.plot( S.i, S.j, 'ok' )
    plt.xlim( -1, Ns )
    plt.ylim( -1, Nt )
    plt.xlabel( 'Source neuron index' )
    plt.ylabel( 'Target neuron index' )
    plt.show()


def make_folders( path ):
    if not os.path.exists( path ):
        os.makedirs( path )


def make_timestamped_folder( path ):
    if not os.path.exists( path ):
        os.makedirs( path )
    timestamp = datetime.datetime.now().strftime( "%Y-%m-%d_%H-%M-%S" )
    os.makedirs( os.path.join( path, timestamp ) )
    
    return os.path.join( path, timestamp )


def contiguous_regions( condition ):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    
    # Find the indices of changes in "condition"
    d = np.diff( condition )
    idx, = d.nonzero()
    
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1
    
    if condition[ 0 ]:
        # If the start of condition is True prepend a 0
        idx = np.r_[ 0, idx ]
    
    if condition[ -1 ]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[ idx, condition.size - 1 ]  # Edit
    
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    
    return idx


def find_ps( path, sim_time, attractor, write_to_file=False ):
    import pyspike as spk
    from scipy.ndimage.filters import uniform_filter1d
    import csv
    
    spike_trains = spk.load_spike_trains_from_txt( os.path.join( path, 'spikes_pyspike.txt' ),
                                                   edges=(0, sim_time),
                                                   ignore_empty_lines=False )
    
    spike_sync_profile = spk.spike_sync_profile( spike_trains, indices=attractor[ 1 ] )
    
    x, y = spike_sync_profile.get_plottable_data()
    # mean_filter_size = round( len( x ) / 10 )
    mean_filter_size = 20
    
    try:
        y_smooth = uniform_filter1d( y, size=mean_filter_size )
    except:
        y_smooth = np.zeros( len( x ) )
    
    pss = contiguous_regions( y_smooth > 0.8 )
    
    if write_to_file:
        if not os.path.exists( os.path.join( path, 'pss.csv' ) ):
            with open( os.path.join( path, 'pss.csv' ), 'w' ) as f:
                writer = csv.writer( f )
                writer.writerow( [ 'Attractor', 'start_s', 'end_s', 'center', 'max', 'mean', 'std' ] )
        
        with open( os.path.join( path, 'pss.csv' ), 'a', newline='' ) as f:
            writer = csv.writer( f )
            
            for ps in pss:
                writer.writerow( [
                        attractor[ 0 ],
                        x[ ps[ 0 ] ],
                        x[ ps[ 1 ] ],
                        x[ ps[ 0 ] + np.argmax( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ],
                        np.max( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ),
                        np.mean( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ),
                        np.std( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ),
                        ] )
    
    return x, y, y_smooth, pss


def read_ps( path, verbose=False ):
    import csv
    
    fn = os.path.join( path, 'pss.csv' )
    with open( fn, 'r' ) as f:
        reader = csv.reader( f )
        
        next( reader, None )
        for line in reader:
            attractor = line[ 0 ]
            start_s = float( line[ 1 ] )
            end_s = float( line[ 2 ] )
            center = float( line[ 3 ] )
            max_ = float( line[ 4 ] )
            mean = float( line[ 5 ] )
            std = float( line[ 6 ] )
            
            if verbose:
                print( f'Found PS in {attractor} '
                       f'between {start_s} s and {end_s} s '
                       f'centered at {center} s '
                       f'with max value {max_} '
                       f'(mean={mean}, '
                       f'std={std})'
                       )
    
    with open( fn ) as f:
        num_rows = sum( 1 for line in f )
    num_rows -= 1
    
    print( 'Found ', num_rows, 'PSs' )
    
    return num_rows
