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


def plot_SPIKE_profile( rcn, filename=None, sim_time=6 ):
    import matplotlib.pyplot as plt
    import os
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
    fig = plt.figure( figsize=(65, 45) )
    gs = plt.GridSpec( 3, 2, width_ratios=[ 4, 1 ] )
    plt.rcParams.update( { 'font.size': 44 } )
    
    # -- plot neuronal spikes with attractors in different colours
    ax_spikes = fig.add_subplot( gs[ 0, 0 ] )
    A1_indexes = np.argwhere( np.logical_and( ids >= A1[ 0 ], ids <= A1[ -1 ] ) )
    A2_indexes = np.argwhere( np.logical_and( ids >= A2[ 0 ], ids <= A2[ -1 ] ) )
    A1_spks = spks[ A1_indexes ]
    A2_spks = spks[ A2_indexes ]
    A1_ids = ids[ A1_indexes ]
    A2_ids = ids[ A2_indexes ]
    ax_spikes.plot( A1_spks, A1_ids, '.', color='orange', label='A1' )
    ax_spikes.plot( A2_spks, A2_ids, '.', color='green', label='A2' )
    ax_spikes.set_xlim( 0, sim_time )
    ax_spikes.set_ylim( 0, rcn.N_input_e )
    ax_spikes.set_xlabel( 'Time (s)' )
    ax_spikes.set_ylabel( 'Neuron ID' )
    ax_spikes.set_title( 'Neuronal spikes' )
    ax_spikes.legend()
    
    # -- plot spike distance profile
    ax_distance = fig.add_subplot( gs[ 1, 0 ] )
    ax_distance.plot( *spike_distance_profile_A1.get_plottable_data(), '--', color='orange', label='A1' )
    ax_distance.plot( *spike_distance_profile_A2.get_plottable_data(), '--', color='green', label='A2' )
    ax_distance.set_xlim( 0, sim_time )
    ax_distance.set_ylim( 0, 1 )
    ax_distance.set_xlabel( 'Time (s)' )
    ax_distance.set_ylabel( 'SPIKE-distance' )
    ax_distance.set_title( 'SPIKE-distance profile' )
    ax_distance.legend()
    
    ax_distance_matrix = fig.add_subplot( gs[ 1, 1 ] )
    spike_distance = spk.spike_distance_matrix( spike_trains )
    im1 = ax_distance_matrix.imshow( spike_distance, vmin=0, vmax=1, interpolation='none' )
    # ax_distance_matrix.set_clim( 0, 1 )
    fig.colorbar( im1, ax=ax_distance_matrix )
    ax_distance_matrix.set_title( "SPIKE-distance matrix" )
    
    # -- plot spike sync profile
    ax_sync = fig.add_subplot( gs[ 2, 0 ] )
    ax_sync.plot( *spike_sync_profile_A1.get_plottable_data(), '--', color='orange', label='A1' )
    ax_sync.plot( *spike_sync_profile_A2.get_plottable_data(), '--', color='green', label='A2' )
    ax_sync.set_xlim( 0, sim_time )
    ax_sync.set_ylim( 0, 1 )
    ax_sync.set_xlabel( 'Time (s)' )
    ax_sync.set_ylabel( 'SPIKE-sync' )
    ax_sync.set_title( 'SPIKE-sync profile' )
    ax_sync.legend()
    
    ax_distance_matrix = fig.add_subplot( gs[ 2, 1 ] )
    spike_distance = spk.spike_sync_matrix( spike_trains )
    im1 = ax_distance_matrix.imshow( spike_distance, vmin=0, vmax=1, interpolation='none' )
    # ax_distance_matrix.set_clim( 0, 1 )
    fig.colorbar( im1, ax=ax_distance_matrix )
    ax_distance_matrix.set_title( "SPIKE-sync matrix" )
    
    # -- figure adjustments
    plt.tight_layout()
    
    fig.show()
    fig.savefig( './attractor_synchronisation.pdf',
                 bbox_inches='tight' )
    
    print( "Average SPIKE-distance within A1:", spike_distance_profile_A1.avrg() )
    print( "Average SPIKE-distance within A2:", spike_distance_profile_A2.avrg() )
    print( "Average SPIKE-synchronisation within A1:", spike_sync_profile_A1.avrg() )
    print( "Average SPIKE-synchronisation within A2:", spike_sync_profile_A2.avrg() )
