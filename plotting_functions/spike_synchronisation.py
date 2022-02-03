import termios

import numpy as np


def save_spikes_pyspike( spikes ):
    filename = '../graph_analysis/E_spikes.txt'
    
    with open( filename, 'w' ) as file:
        file.write( '# Spikes from excitatory population' )
        for v in spikes.values():
            np.savetxt( file, np.array( v ), newline=' ' )
            file.write( '\n' )
    
    return filename


def plot_SPIKE_profile( rcn, filename=None, sim_time=6 ):
    import matplotlib.pyplot as plt
    import os
    import pyspike as spk
    
    plt.rcParams.update( { 'font.size': 22 } )
    
    # -- write spikes to text file if it is not supplied or it does not exist
    if not filename or not os.path.exists( filename ):
        filename = save_spikes_pyspike( rcn.get_E_spks( spike_trains=True ) )
    spks, ids = rcn.get_E_spks()
    spike_trains = spk.load_spike_trains_from_txt( filename, edges=(0, sim_time),
                                                   ignore_empty_lines=False )
    
    # -- define attractor indices
    A1 = (0, 64)
    A2 = (100, 164)
    
    # -- compute distance and sync profiles for the two attractors
    spike_distance_profile_A1 = spk.spike_profile( spike_trains, indices=A1 )
    spike_distance_profile_A2 = spk.spike_profile( spike_trains, indices=A2 )
    spike_sync_profile_A1 = spk.spike_sync_profile( spike_trains, indices=A1 )
    spike_sync_profile_A2 = spk.spike_sync_profile( spike_trains, indices=A2 )
    
    # -- make figure
    fig = plt.figure( figsize=(30, 20) )
    
    # -- plot neuronal spikes with attractors in different colours
    ax_spikes = fig.add_subplot( 3, 1, 1 )
    A1_indexes = np.argwhere( np.logical_and( ids >= A1[ 0 ], ids < A1[ 1 ] ) )
    A2_indexes = np.argwhere( np.logical_and( ids >= A2[ 0 ], ids < A2[ 1 ] ) )
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
    ax_distance = fig.add_subplot( 3, 1, 2 )
    ax_distance.plot( *spike_distance_profile_A1.get_plottable_data(), '--', color='orange', label='A1' )
    ax_distance.plot( *spike_distance_profile_A2.get_plottable_data(), '--', color='green', label='A2' )
    ax_distance.set_xlim( 0, sim_time )
    ax_distance.set_ylim( 0, 1 )
    ax_distance.set_xlabel( 'Time (s)' )
    ax_distance.set_ylabel( 'SPIKE-distance' )
    ax_distance.set_title( 'Spike distance profile' )
    ax_distance.legend()
    
    # -- plot spike sync profile
    ax_sync = fig.add_subplot( 3, 1, 3 )
    ax_sync.plot( *spike_sync_profile_A1.get_plottable_data(), '--', color='orange', label='A1' )
    ax_sync.plot( *spike_sync_profile_A2.get_plottable_data(), '--', color='green', label='A2' )
    ax_sync.set_xlim( 0, sim_time )
    ax_sync.set_ylim( 0, 1 )
    ax_sync.set_xlabel( 'Time (s)' )
    ax_sync.set_ylabel( 'SPIKE-syncronisation' )
    ax_sync.set_title( 'Spike sync profile' )
    ax_sync.legend()
    
    # -- figure adjustments
    plt.tight_layout()
    
    fig.show()
    fig.savefig( './attractor_synchronisation.png',
                 bbox_inches='tight' )
    
    print( "SPIKE distance A1:", spike_distance_profile_A1.avrg() )
    print( "SPIKE distance A2:", spike_distance_profile_A2.avrg() )
    print( "SPIKE synchronisation A1:", spike_sync_profile_A1.avrg() )
    print( "SPIKE synchronisation A2:", spike_sync_profile_A2.avrg() )
