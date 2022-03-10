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


def find_ps( path, sim_time, attractor, write_to_file=False, ba=None, gs=None, verbose=False ):
    import pyspike as spk
    from scipy.ndimage.filters import uniform_filter1d
    import pandas as pd
    
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
        assert ba is not None
        assert gs is not None
        
        df = pd.DataFrame( columns=[ 'atr', 'ba_Hz', 'gs_%', 'start_s', 'end_s', 'center', 'max', 'mean', 'std' ] )
        
        for ps in pss:
            df = df.append( pd.DataFrame( [ [ attractor[ 0 ],
                                              ba,
                                              gs[ 0 ][ 0 ],
                                              x[ ps[ 0 ] ],
                                              x[ ps[ 1 ] ],
                                              x[ ps[ 0 ] + np.argmax( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ],
                                              np.max( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ),
                                              np.mean( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ),
                                              np.std( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ] ],
                                          columns=[ 'atr', 'ba_Hz', 'gs_%', 'start_s', 'end_s', 'center', 'max', 'mean',
                                                    'std' ] ) )
        
        fn = os.path.join( path, "pss.xlsx" )
        if not os.path.isfile( fn ):
            with pd.ExcelWriter( fn ) as writer:
                df.to_excel( writer, index=False )
        else:
            append_df_to_excel( df, fn )
    
    if verbose:
        print( f'Found PS in {attractor[ 0 ]} '
               f'between {x[ ps[ 0 ] ]} s and {x[ ps[ 1 ] ]} s '
               f'centered at {x[ ps[ 0 ] + np.argmax( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ]} s '
               f'with max value {np.max( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )} '
               f'(mean={np.mean( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )}, '
               f'std={np.std( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )}'
               )
    
    return x, y, y_smooth, pss


# TODo save these somewhere sensible
def count_pss_in_gss( path, gss ):
    import pandas as pd
    
    df = pd.read_excel( os.path.join( path, 'pss.xlsx' ) )
    
    num_ps_in_gs = 0
    for _, row in df.iterrows():
        for gs in gss:
            if row[ 'start_s' ] >= gs[ 1 ][ 0 ] and row[ 'end_s' ] <= gs[ 1 ][ 1 ]:
                num_ps_in_gs += 1
    
    total_num_ps = len( df )
    print( f'Found {num_ps_in_gs} PS in GS out of {total_num_ps} total PS' )


def append_pss_to_xlsx( experiment_path, iteration_path ):
    import pandas as pd
    
    df_iteration = pd.read_excel( os.path.join( iteration_path, 'pss.xlsx' ) )
    
    fn = os.path.join( experiment_path, 'pss.xlsx' )
    if os.path.isfile( fn ):
        df_experiment = pd.read_excel( fn )
        append_df_to_excel( df_iteration, fn )
    else:
        with pd.ExcelWriter( fn ) as writer:
            df_iteration.to_excel( writer, index=False )


def append_df_to_excel( df, excel_path ):
    import pandas as pd
    
    df_excel = pd.read_excel( excel_path )
    result = pd.concat( [ df_excel, df ], ignore_index=True )
    result.to_excel( excel_path, index=False )


def compute_pss_statistics( timestamp_folder, generic_stimuli=False ):
    import pandas as pd
    
    fn = os.path.join( timestamp_folder, 'pss.xlsx' )
    df = pd.read_excel( fn )
    
    return df


def generate_gss( gs_percentage, gs_freq, gs_length, pre_runtime, gs_runtime ):
    free_time = gs_runtime - gs_freq * gs_length
    free_time /= (gs_freq - 1)
    
    gss_times = [ ]
    for t in np.arange( pre_runtime, pre_runtime + gs_runtime, free_time + gs_length ):
        gss_times.append( (gs_percentage, (t, t + gs_length)) )
    
    return gss_times, free_time
