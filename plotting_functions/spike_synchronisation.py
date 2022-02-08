import termios

import numpy as np


def save_spikes_pyspike( spikes ):
    filename = '../graph_analysis/E_spikes.txt'
    
    with open( filename, 'w' ) as file:
        file.write( '# Spikes from excitatory population\n\n' )
        for v in spikes.values():
            np.savetxt( file, np.array( v ), newline=' ' )
            file.write( '\n' )
    
    return filename


def video_SPIKE_profile( rcn, filename=None, sim_time=6 ):
    pass


# TODO measure algebraic connectivity at regular intervals and add it to plots
# TODO look at sync in inhibitory neurons
# TODO count de-synchronised neurons in each attractor
def plot_SPIKE_profile( rcn, filename=None, sim_time=6 ):
    import os
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import uniform_filter1d
    import pyspike as spk
    
    # -- write spikes to text file if it is not supplied or it does not exist
    if not filename or not os.path.exists( filename ):
        filename = save_spikes_pyspike( rcn.get_E_spks( spike_trains=True ) )
    spks, ids = rcn.get_E_spks()
    spike_trains = spk.load_spike_trains_from_txt( filename, edges=(0, sim_time),
                                                   ignore_empty_lines=False )
    
    # -- define attractor indices
    A1 = list( range( 0, 64 ) )
    A2 = list( range( 100, 164 ) )
    
    # -- compute distance and sync profiles for the two attractors
    spike_distance_profile_A1 = spk.spike_profile( spike_trains, indices=A1 )
    spike_distance_profile_A2 = spk.spike_profile( spike_trains, indices=A2 )
    spike_sync_profile_A1 = spk.spike_sync_profile( spike_trains, indices=A1 )
    spike_sync_profile_A2 = spk.spike_sync_profile( spike_trains, indices=A2 )
    
    # -- make figure
    fig = plt.figure( figsize=(30, 20) )
    gs = plt.GridSpec( 3, 2, width_ratios=[ 4, 1 ] )
    plt.rcParams.update( { 'font.size': 22 } )
    mean_filter_size = 500
    
    # -- plot neuronal spikes with attractors in different colours
    ax_spikes = fig.add_subplot( gs[ 0, 0 ] )
    A1_indexes = np.argwhere( np.logical_and( ids >= A1[ 0 ], ids <= A1[ -1 ] ) )
    A2_indexes = np.argwhere( np.logical_and( ids >= A2[ 0 ], ids <= A2[ -1 ] ) )
    A1_spks = spks[ A1_indexes ]
    A2_spks = spks[ A2_indexes ]
    A1_ids = ids[ A1_indexes ]
    A2_ids = ids[ A2_indexes ]
    ax_spikes.plot( A1_spks, A1_ids, '.', markersize=0.5, alpha=0.5, color='orange', label='A1' )
    ax_spikes.plot( A2_spks, A2_ids, '.', markersize=0.5, alpha=0.5, color='green', label='A2' )
    ax_spikes.set_xlim( 0, sim_time )
    ax_spikes.set_ylim( 0, rcn.N_input_e )
    ax_spikes.set_xlabel( 'Time (s)', labelpad=10 )
    ax_spikes.set_ylabel( 'Neuron ID' )
    ax_spikes.set_title( 'Neuronal spikes' )
    ax_spikes.legend()
    
    # -- add stimulus presentation markers
    trans = ax_spikes.get_xaxis_transform()
    ax_spikes.annotate( 'Stimulus 1', xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_spikes.plot( [ 0.1, 1.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_spikes.annotate( 'Delay 1', xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_spikes.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_spikes.annotate( 'Stimulus 2', xy=(4, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_spikes.plot( [ 3.1, 4.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    ax_spikes.annotate( 'Delay 2', xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_spikes.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    
    # -- plot spike distance profile
    ax_distance = fig.add_subplot( gs[ 1, 0 ] )
    x, y = spike_distance_profile_A1.get_plottable_data()
    ax_distance.plot( x, y, '--', color='orange', label='A1' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_distance.plot( x, y_smooth, '.', markersize=0.5, color='orange' )
    x, y = spike_distance_profile_A2.get_plottable_data()
    ax_distance.plot( x, y, '--', color='green', label='A2' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_distance.plot( x, y_smooth, '.', markersize=0.5, color='green' )
    ax_distance.set_xlim( 0, sim_time )
    ax_distance.set_ylim( 0, 1 )
    ax_distance.set_xlabel( 'Time (s)', labelpad=10 )
    ax_distance.set_ylabel( 'SPIKE-distance' )
    ax_distance.set_title( 'SPIKE-distance profile' )
    ax_distance.legend()
    
    # -- add average measure markers
    trans = ax_distance.get_xaxis_transform()
    ax_distance.annotate( f'{spk.spike_distance( spike_trains, indices=A1, interval=(0, 2) ):.5f}',
                          xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_distance.plot( [ 0, 2 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_distance.annotate( f'{spk.spike_distance( spike_trains, indices=A1, interval=(2, 3) ):.5f}',
                          xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_distance.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_distance.annotate( f'{spk.spike_distance( spike_trains, indices=A2, interval=(3, 5) ):.5f}',
                          xy=(4, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_distance.plot( [ 3, 5 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    ax_distance.annotate( f'{spk.spike_distance( spike_trains, indices=A2, interval=(5, 6) ):.5f}',
                          xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_distance.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    
    ax_distance_matrix = fig.add_subplot( gs[ 1, 1 ] )
    spike_distance = spk.spike_distance_matrix( spike_trains )
    im1 = ax_distance_matrix.imshow( spike_distance, vmin=0, vmax=1, interpolation='none' )
    # ax_distance_matrix.set_clim( 0, 1 )
    fig.colorbar( im1, ax=ax_distance_matrix )
    ax_distance_matrix.set_title( "SPIKE-distance matrix" )
    
    # -- plot spike sync profile
    ax_sync = fig.add_subplot( gs[ 2, 0 ] )
    x, y = spike_sync_profile_A1.get_plottable_data()
    ax_sync.plot( x, y, '--', color='orange', alpha=0.5, label='A1' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_sync.plot( x, y_smooth, '.', markersize=0.5, color='orange' )
    x, y = spike_sync_profile_A2.get_plottable_data()
    ax_sync.plot( x, y, '--', color='green', alpha=0.5, label='A2' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_sync.plot( x, y_smooth, '.', markersize=0.5, color='green' )
    ax_sync.set_xlim( 0, sim_time )
    ax_sync.set_ylim( 0, 1 )
    ax_sync.set_xlabel( 'Time (s)', labelpad=10 )
    ax_sync.set_ylabel( 'SPIKE-sync' )
    ax_sync.set_title( 'SPIKE-sync profile' )
    ax_sync.legend()
    
    # -- add average measure markers
    trans = ax_sync.get_xaxis_transform()
    ax_sync.annotate( f'{spk.spike_sync( spike_trains, indices=A1, interval=(0, 2) ):.5f}',
                      xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_sync.plot( [ 0, 2 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_sync.annotate( f'{spk.spike_sync( spike_trains, indices=A1, interval=(2, 3) ):.5f}',
                      xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax_sync.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax_sync.annotate( f'{spk.spike_sync( spike_trains, indices=A2, interval=(3, 5) ):.5f}',
                      xy=(4, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_sync.plot( [ 3, 5 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    ax_sync.annotate( f'{spk.spike_sync( spike_trains, indices=A2, interval=(5, 6) ):.5f}',
                      xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax_sync.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    
    ax_sync_matrix = fig.add_subplot( gs[ 2, 1 ] )
    spike_sync = spk.spike_sync_matrix( spike_trains )
    im1 = ax_sync_matrix.imshow( spike_sync, vmin=0, vmax=1, interpolation='none' )
    # ax_sync_matrix.set_clim( 0, 1 )
    fig.colorbar( im1, ax=ax_sync_matrix )
    ax_sync_matrix.set_title( "SPIKE-sync matrix" )
    
    # -- figure adjustments
    plt.tight_layout()
    
    fig.show()
    fig.savefig( './attractor_synchronisation.pdf',
                 bbox_inches='tight' )
    
    # print( "Average SPIKE-distance within A1 (0, 2)s:",
    #        spk.spike_distance( spike_trains, indices=A1, interval=(0, 2) ) )
    # print( "Average SPIKE-distance within A1 (2, 3)s:",
    #        spk.spike_distance( spike_trains, indices=A1, interval=(2, 3) ) )
    # print( "Average SPIKE-distance within A2 (3, 5)s:",
    #        spk.spike_distance( spike_trains, indices=A2, interval=(3, 5) ) )
    # print( "Average SPIKE-distance within A2 (5, 6)s:",
    #        spk.spike_distance( spike_trains, indices=A2, interval=(5, 6) ) )
    # print( "Average SPIKE-sync within A1 (0, 2)s:", spk.spike_sync( spike_trains, indices=A1, interval=(0, 2) ) )
    # print( "Average SPIKE-sync within A1 (2, 3)s:", spk.spike_sync( spike_trains, indices=A1, interval=(2, 3) ) )
    # print( "Average SPIKE-sync within A2 (3, 5)s:", spk.spike_sync( spike_trains, indices=A2, interval=(3, 5) ) )
    # print( "Average SPIKE-sync within A2 (5, 6)s:", spk.spike_sync( spike_trains, indices=A2, interval=(5, 6) ) )
