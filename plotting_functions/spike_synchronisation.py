import pathlib

import numpy as np

from brian2 import second, Quantity
import pyspike as spk


def save_spikes_pyspike( spikes, filename ):
    with open( filename, 'w' ) as file:
        file.write( '# Spikes from excitatory population\n\n' )
        for v in spikes.values():
            np.savetxt( file, np.array( v ), newline=' ' )
            file.write( '\n' )
    
    return filename


def video_attractor_profile( rcn, sim_time=6 * second, gather_every=0.1 * second ):
    import tempfile
    from tqdm import tqdm
    import os
    import math
    import pandas as pd
    
    if not isinstance( gather_every, Quantity ):
        gather_every *= second
    
    # -- read min and max algebraic connectivity to fix it in the video from the start
    df = pd.read_csv( './a_conn.csv', dtype=np.float64 )
    min_a_conn = df[ [ 'A1', 'A2' ] ].min().min()
    max_a_conn = df[ [ 'A1', 'A2' ] ].max().max()
    
    temp_dir = tempfile.TemporaryDirectory()
    time_points = np.arange( 0, sim_time + gather_every, gather_every )
    pad = math.floor( math.log10( len( time_points ) ) ) + 1
    for i, t in tqdm( enumerate( time_points ), desc='Creating video frames',
                      total=int( sim_time / gather_every ), unit='frame' ):
        temp_file = os.path.join( temp_dir.name, f'{i:0{pad}}' )
        plot_attractor_profile( rcn, filename=temp_file, show=False, curr_time=t,
                                fig_width=30, y_lim=(min_a_conn, max_a_conn) )
    
    ffmpeg_string = "ffmpeg " + "-pattern_type glob -i '" \
                    + temp_dir.name + "/" + "*.png' " \
                                            "-c:v libx264 -preset veryslow -crf 17 " \
                                            "-tune stillimage -hide_banner -loglevel warning " \
                                            "-y -pix_fmt yuv420p " \
                                            "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' " \
                    + "./attractor_video.mp4"
    
    print( f"Generating Video using {ffmpeg_string} ..." )
    os.system( ffmpeg_string )
    print( f"Generated video {f'{os.getcwd()}/attractor_video.mp4'}" )
    temp_dir.cleanup()


def _plot_matrix( fig, ax, spike_trains, measure, interval=None ):
    interval = tuple( np.clip( interval, 0, np.Inf ) )
    if interval[ 1 ] == 0:
        interval = None
    
    if measure == 'distance':
        spike_profile = spk.spike_distance_matrix( spike_trains, interval=interval )
    elif measure == 'sync':
        spike_profile = spk.spike_sync_matrix( spike_trains, interval=interval )
    else:
        raise ValueError( 'title must be distance or sync' )
    
    ax_matrix = fig.add_subplot( ax )
    im = ax_matrix.imshow( spike_profile, vmin=0, vmax=1, interpolation='none' )
    # ax_distance_matrix.set_clim( 0, 1 )
    fig.colorbar( im, ax=ax_matrix )
    if interval:
        ax_matrix.set_title( f"SPIKE-{measure} matrix\n[{interval[ 0 ]:.1f}-{interval[ 1 ]:.1f}] s" )
    else:
        ax_matrix.set_title( f"SPIKE-{measure} matrix\n[0.0-0.0] s" )
    
    return ax_matrix


def _subaxis_mean_annotations( fig, ax, input, indices, measure ):
    import pandas as pd
    
    if measure == 'distance':
        spike_profile = spk.spike_distance
    elif measure == 'sync':
        spike_profile = spk.spike_sync
    elif measure == 'connectivity':
        pass
    else:
        raise ValueError( 'title must be distance, sync, or connectivity' )
    
    A1 = indices[ 0 ]
    A2 = indices[ 1 ]
    
    trans = ax.get_xaxis_transform()
    
    # -- line markers
    ax.plot( [ 0.1, 1.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax.plot( [ 3.1, 4.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    
    # -- data
    if isinstance( input, pd.DataFrame ) and measure == 'connectivity':
        ax.plot( [ 0.1, 1.9 ], [ -.2, -.2 ], linewidth=5, color="green", transform=trans, clip_on=False )
        
        ax.plot( [ 2.1, 2.9 ], [ -.2, -.2 ], linewidth=5, color="green", transform=trans, clip_on=False )
        
        ax.plot( [ 3.1, 4.9 ], [ -.2, -.2 ], linewidth=5, color="green", transform=trans, clip_on=False )
        
        ax.plot( [ 5.1, 5.9 ], [ -.2, -.2 ], linewidth=5, color="green", transform=trans, clip_on=False )
        
        # -- ... for stimulus 1
        ax.annotate( f'{input.loc[ :2 ][ "A1" ].mean():.5f}',
                     xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.annotate( f'{input.loc[ :2 ][ "A2" ].mean():.5f}',
                     xy=(1, -.22), xycoords=trans, ha="center", va="top", color='green' )
        # -- delay 1
        ax.annotate( f'{input.loc[ 2:3 ][ "A1" ].mean():.5f}',
                     xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.annotate( f'{input.loc[ 2:3 ][ "A2" ].mean():.5f}',
                     xy=(2.5, -.22), xycoords=trans, ha="center", va="top", color='green' )
        # -- stimulus 2
        ax.annotate( f'{input.loc[ 3:5 ][ "A1" ].mean():.5f}',
                     xy=(4, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.annotate( f'{input.loc[ 3:5 ][ "A2" ].mean():.5f}',
                     xy=(4, -.22), xycoords=trans, ha="center", va="top", color='green' )
        # -- delay 2
        ax.annotate( f'{input.loc[ 5: ][ "A1" ].mean():.5f}',
                     xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.annotate( f'{input.loc[ 5: ][ "A2" ].mean():.5f}',
                     xy=(5.5, -.22), xycoords=trans, ha="center", va="top", color='green' )
    else:
        ax.annotate( f'{spike_profile( input, indices=A1, interval=(0, 2) ):.5f}',
                     xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.plot( [ 0.1, 1.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
        ax.annotate( f'{spike_profile( input, indices=A1, interval=(2, 3) ):.5f}',
                     xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
        ax.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
        ax.annotate( f'{spike_profile( input, indices=A2, interval=(3, 5) ):.5f}',
                     xy=(4, -.1), xycoords=trans, ha="center", va="top", color='green' )
        ax.plot( [ 3.1, 4.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
        ax.annotate( f'{spike_profile( input, indices=A2, interval=(5, 6) ):.5f}',
                     xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='green' )
        ax.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )


def _subaxis_annotations( fig, ax ):
    trans = ax.get_xaxis_transform()
    ax.annotate( 'Stimulus 1',
                 xy=(1, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax.plot( [ 0.1, 1.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax.annotate( 'Delay 1',
                 xy=(2.5, -.1), xycoords=trans, ha="center", va="top", color='orange' )
    ax.plot( [ 2.1, 2.9 ], [ -.08, -.08 ], linewidth=5, color="orange", transform=trans, clip_on=False )
    ax.annotate( 'Stimulus 2',
                 xy=(4, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax.plot( [ 3.1, 4.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )
    ax.annotate( 'Delay 2',
                 xy=(5.5, -.1), xycoords=trans, ha="center", va="top", color='green' )
    ax.plot( [ 5.1, 5.9 ], [ -.08, -.08 ], linewidth=5, color="green", transform=trans, clip_on=False )


# TODO look at sync in inhibitory neurons
# TODO count de-synchronised neurons in each attractor
# TODO check if video exists to generate multiple videos
# TODO make it independent of RCN!!
def plot_attractor_profile( rcn, filename='attractor_profile.png', show=True, spikes_filename=None,
                            sim_time=6, curr_time=None,
                            fig_width=30, y_lim=None ):
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import uniform_filter1d
    import pandas as pd
    
    if curr_time is None:
        curr_time = sim_time
    
    assert not (spikes_filename is None and curr_time is None), 'If you provide a spikes filename, you must not ' \
                                                                'provide a current time'
    
    if spikes_filename:
        spike_trains = spk.load_spike_trains_from_txt( spikes_filename, edges=(0, sim_time),
                                                       ignore_empty_lines=False )
    else:
        spike_trains = [ ]
        for v in rcn.get_E_spks( spike_trains=True ).values():
            # -- filter spikes happening after curr_time
            spikes = v[ v < curr_time * second ]
            spike_trains.append( spk.SpikeTrain( spikes, [ 0, curr_time ] ) )
    
    spks, ids = rcn.get_E_spks()
    # -- also remove spikes after curr_time
    time_mask = spks < curr_time * second
    spks = spks[ time_mask ]
    ids = ids[ time_mask ]
    
    # -- define attractor indices
    A1 = list( range( 0, 64 ) )
    A2 = list( range( 100, 164 ) )
    
    # -- compute distance and sync profiles for the two attractors
    spike_distance_profile_A1 = spk.spike_profile( spike_trains, indices=A1 )
    spike_distance_profile_A2 = spk.spike_profile( spike_trains, indices=A2 )
    spike_sync_profile_A1 = spk.spike_sync_profile( spike_trains, indices=A1 )
    spike_sync_profile_A2 = spk.spike_sync_profile( spike_trains, indices=A2 )
    
    # -- make figure
    fig_height = fig_width / 1.4
    fig = plt.figure( figsize=(fig_width, fig_height), dpi=100 )
    gs = plt.GridSpec( 4, 2, width_ratios=[ 4, 1 ] )
    plt.rcParams.update( { 'font.size': fig_height } )
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
    ax_spikes.set_yticks( [ 0, 64, 100, 164 ] )
    ax_spikes.set_title( 'Neuronal spikes' )
    ax_spikes.legend( loc='upper right' )
    
    # -- add stimulus presentation markers
    if curr_time == sim_time:
        _subaxis_annotations( fig, ax_spikes )
    
    # -- plot spike distance profile
    ax_distance = fig.add_subplot( gs[ 1, 0 ] )
    x, y = spike_distance_profile_A1.get_plottable_data()
    ax_distance.plot( x, y, color='orange', label='A1' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_distance.plot( x, y_smooth, '.', markersize=0.5, color='orange' )
    x, y = spike_distance_profile_A2.get_plottable_data()
    ax_distance.plot( x, y, color='green', label='A2' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_distance.plot( x, y_smooth, '.', markersize=0.5, color='green' )
    ax_distance.set_xlim( 0, sim_time )
    ax_distance.set_ylim( 0, 1 )
    ax_distance.set_xlabel( 'Time (s)', labelpad=10 )
    ax_distance.set_ylabel( 'SPIKE-distance' )
    ax_distance.set_title( 'SPIKE-distance profile' )
    ax_distance.legend( loc='upper right' )
    
    # -- add average measure markers
    if curr_time == sim_time:
        _subaxis_mean_annotations( fig, ax_distance, spike_trains, (A1, A2), measure='distance' )
        _plot_matrix( fig, gs[ 1, 1 ], spike_trains, 'distance', interval=(0, curr_time) )
    else:
        _plot_matrix( fig, gs[ 1, 1 ], spike_trains, 'distance', interval=(curr_time - 0.1, curr_time) )
    
    # -- plot spike sync profile
    ax_sync = fig.add_subplot( gs[ 2, 0 ] )
    x, y = spike_sync_profile_A1.get_plottable_data()
    ax_sync.plot( x, y, color='orange', alpha=0.5, label='A1' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_sync.plot( x, y_smooth, '.', markersize=0.5, color='orange' )
    x, y = spike_sync_profile_A2.get_plottable_data()
    ax_sync.plot( x, y, color='green', alpha=0.5, label='A2' )
    y_smooth = uniform_filter1d( y, size=mean_filter_size )
    ax_sync.plot( x, y_smooth, '.', markersize=0.5, color='green' )
    ax_sync.set_xlim( 0, sim_time )
    ax_sync.set_ylim( 0, 1 )
    ax_sync.set_xlabel( 'Time (s)', labelpad=10 )
    ax_sync.set_ylabel( 'SPIKE-sync' )
    ax_sync.set_title( 'SPIKE-sync profile' )
    ax_sync.legend( loc='upper right' )
    
    # -- add average measure markers
    if curr_time == sim_time:
        _subaxis_mean_annotations( fig, ax_sync, spike_trains, (A1, A2), measure='sync' )
        _plot_matrix( fig, gs[ 2, 1 ], spike_trains, 'sync', interval=(0, curr_time) )
    else:
        _plot_matrix( fig, gs[ 2, 1 ], spike_trains, 'sync', interval=(curr_time - 0.1, curr_time) )
    
    # -- plot algebraic connectivity
    df = pd.read_csv( './a_conn.csv', dtype=np.float64 )
    df.set_index( 't', inplace=True )
    
    ax_a_conn = fig.add_subplot( gs[ 3, 0 ] )
    df.loc[ :curr_time ][ 'A1' ].plot( ax=ax_a_conn, color='orange', label='A1' )
    df.loc[ :curr_time ][ 'A2' ].plot( ax=ax_a_conn, color='green', label='A2' )
    ax_a_conn.set_xlim( 0, sim_time )
    ax_a_conn.set_ylim( y_lim )
    ax_a_conn.set_xlabel( 'Time (s)', labelpad=20 )
    ax_a_conn.set_ylabel( r'$\lambda_2$' )
    ax_a_conn.set_title( 'Algebraic connectivity profile' )
    ax_a_conn.legend( loc='upper right' )
    
    # -- add average measure markers ...
    if curr_time == sim_time:
        _subaxis_mean_annotations( fig, ax_a_conn, df, (A1, A2), measure='connectivity' )
    
    # -- add vertical lines to mark where we've set curr_time
    if curr_time != sim_time:
        for ax in [ ax_spikes, ax_distance, ax_sync, ax_a_conn ]:
            ax.axvline( curr_time, color='black', linewidth=1 )
    
    # -- figure adjustments
    plt.tight_layout()
    
    if show:
        fig.show()
    
    # -- check that filename ends with .png
    if filename:
        filename_stripped = pathlib.Path( filename )
        if not filename_stripped.suffix == '.png':
            filename = filename_stripped.with_suffix( '.png' )
            fig.savefig( f'{filename}.png', bbox_inches='tight' )
    
    plt.close( fig )
