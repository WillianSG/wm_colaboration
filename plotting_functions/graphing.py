import networkx as nx
import numpy as np


def rgb_to_hex( r, g, b ):
    return '#{:02X}{:02X}{:02X}'.format( r, g, b )


def hex_to_rgb( h ):
    h = h.lstrip( '#' )
    
    return tuple( int( h[ i:i + 2 ], 16 ) for i in (0, 2, 4) )


def rcn2nx( rcn, neurons_subsample=None, subsample_attractors=False, seed=None,
            remove_edges_threshold=0.0, output_filename='graph' ):
    """Given an instance of a RecurrentCompetitiveNet builds the corresponding NetworkX graph.
    
    Parameters:
    net (RecurrentCompetitiveNet): the Brian2 net of which to build the graph
    neurons_subsample (tuple): number of neurons to sample from excitatory and inhibitory population
    subsample_attractors (bool): ad-hoc sampling from attractor A, attractor B, and non-specific population
    seed (int): set seed to make the subsampling reproducible
    remove_edges_threshold (float): remove edges whose weight is smaller than remove_edges_threshold from graph
    output_filename (str): the name of the file to save to disk in os.getcwd()
    
    Returns:
    networkx.DiGraph: The graph build from net
    
    """
    import os
    
    # ------ Subsample graph nodes
    if neurons_subsample:
        e_neurons_subsample = np.clip( neurons_subsample[ 0 ], 0, len( rcn.E ) )
        i_neurons_subsample = np.clip( neurons_subsample[ 1 ], 0, len( rcn.I ) )
        
        rng = np.random.RandomState( seed )
        if subsample_attractors:
            assert e_neurons_subsample % 3 == 0
            e_neurons = np.concatenate(
                    [ rng.choice( range( 0, 64 ), int( e_neurons_subsample / 3 ), replace=False ),
                      rng.choice( range( 100, 164 ), int( e_neurons_subsample / 3 ), replace=False ),
                      rng.choice( range( 164, 256 ), int( e_neurons_subsample / 3 ), replace=False ) ]
                    )
            i_neurons = rng.choice( range( 0, len( rcn.I ) ), i_neurons_subsample, replace=False )
        else:
            e_neurons = rng.choice( range( 0, len( rcn.E ) ), e_neurons_subsample, replace=False )
            i_neurons = rng.choice( range( 0, len( rcn.I ) ), i_neurons_subsample, replace=False )
    else:
        e_neurons = np.array( range( 0, len( rcn.E ) ) )
        i_neurons = np.array( range( 0, len( rcn.I ) ) )
    
    # ------- Make sure that there are no duplicates in neurons subsample or Brian2 indexing runs into a bug
    assert len( np.unique( e_neurons ) ) == len( e_neurons ) or len( np.unique( i_neurons ) ) == len( i_neurons )
    # ----- Add Excitatory-to-Excitatory connections
    e2e_edges_pre = rcn.E_E.i[ e_neurons, e_neurons ].tolist()
    e2e_edges_post = rcn.E_E.j[ e_neurons, e_neurons ].tolist()
    e2e_edge_weights = rcn.E_E.w_[ e_neurons, e_neurons ].tolist()
    # ----- Add Inhibitory-to-Excitatory connections
    i2e_edges_pre = rcn.I_E.i[ i_neurons, e_neurons ].tolist()
    i2e_edges_post = rcn.I_E.j[ i_neurons, e_neurons ].tolist()
    i2e_edge_weights = rcn.I_E.w_[ i_neurons, e_neurons ].tolist()
    # ----- Add Excitatory-to-Inhibitory connections
    e2i_edges_pre = rcn.E_I.i[ e_neurons, i_neurons ].tolist()
    e2i_edges_post = rcn.E_I.j[ e_neurons, i_neurons ].tolist()
    e2i_edge_weights = rcn.E_I.w_[ e_neurons, i_neurons ].tolist()
    # ----- Add Inhibitory-to-Inhibitory connections
    i2i_edges_pre = rcn.I_I.i[ i_neurons, i_neurons ].tolist()
    i2i_edges_post = rcn.I_I.j[ i_neurons, i_neurons ].tolist()
    i2i_edge_weights = rcn.I_I.w_[ i_neurons, i_neurons ].tolist()
    
    if remove_edges_threshold is not None:
        e2e_edges_pre = [ k for i, k in enumerate( e2e_edges_pre ) if
                          e2e_edge_weights[ i ] > remove_edges_threshold ]
        e2e_edges_post = [ k for i, k in enumerate( e2e_edges_post ) if
                           e2e_edge_weights[ i ] > remove_edges_threshold ]
        e2e_edge_weights = [ k for i, k in enumerate( e2e_edge_weights ) if
                             e2e_edge_weights[ i ] > remove_edges_threshold ]
        i2e_edges_pre = [ k for i, k in enumerate( i2e_edges_pre ) if
                          i2e_edge_weights[ i ] > remove_edges_threshold ]
        i2e_edges_post = [ k for i, k in enumerate( i2e_edges_post ) if
                           i2e_edge_weights[ i ] > remove_edges_threshold ]
        i2e_edge_weights = [ k for i, k in enumerate( i2e_edge_weights ) if
                             i2e_edge_weights[ i ] > remove_edges_threshold ]
        e2i_edges_pre = [ k for i, k in enumerate( e2i_edges_pre ) if
                          e2i_edge_weights[ i ] > remove_edges_threshold ]
        e2i_edges_post = [ k for i, k in enumerate( e2i_edges_post ) if
                           e2i_edge_weights[ i ] > remove_edges_threshold ]
        e2i_edge_weights = [ k for i, k in enumerate( e2i_edge_weights ) if
                             e2i_edge_weights[ i ] > remove_edges_threshold ]
        i2i_edges_pre = [ k for i, k in enumerate( i2i_edges_pre ) if
                          i2i_edge_weights[ i ] > remove_edges_threshold ]
        i2i_edges_post = [ k for i, k in enumerate( i2i_edges_post ) if
                           i2i_edge_weights[ i ] > remove_edges_threshold ]
        i2i_edge_weights = [ k for i, k in enumerate( i2i_edge_weights ) if
                             i2i_edge_weights[ i ] > remove_edges_threshold ]
    
    g = nx.DiGraph()
    
    # Add excitatory neurons
    e_nodes = [ f'e_{i}' for i in e_neurons ]
    for n in e_nodes:
        g.add_node( n, label=n, color='rgba(0,0,255,1)', title=' ', type='excitatory' )
    for i, j, w in zip( e2e_edges_pre, e2e_edges_post, e2e_edge_weights ):
        g.add_edge( f'e_{i}', f'e_{j}', weight=w )
    
    # Classify excitatory neurons
    tag_weakly_connected_components( g )
    tag_attracting_components( g )
    colour_by_attractor( g )
    
    # Add inhibitory neurons
    i_nodes = [ f'i_{i}' for i in i_neurons ]
    for n in i_nodes:
        g.add_node( n, label=n, color='rgba(255,0,0,0.5)', title=' ', type='inhibitory' )
    for i, j, w in zip( i2e_edges_pre, i2e_edges_post, i2e_edge_weights ):
        g.add_edge( f'i_{i}', f'e_{j}', weight=w )
    for i, j, w in zip( e2i_edges_pre, e2i_edges_post, e2i_edge_weights ):
        g.add_edge( f'e_{i}', f'i_{j}', weight=w )
    
    # GraphML does not support list or any non-primitive type
    if output_filename:
        nx.write_graphml( g, f'{os.getcwd()}/{output_filename}.graphml' )
    
    return g


def nx2pyvis( input, notebook=False, output_filename='graph',
              scale_by='neighbours',
              neuron_types=None, synapse_types=None,
              open_output=True, show_buttons=True, only_physics_buttons=True, window_size=(1000, 1000) ):
    """Given an instance of a Networx.Graph builds the corresponding PyVis graph for display purposes.

    Parameters:
    input (nx.Graph or str): the NetworkX graph to convert to PyVis or the path to a GraphML file to convert
    output_filename (str): the name of the file HTML to save to disk in os.getcwd()
    scale_by (str): the node parameter whose value is used to visually scale the nodes.  Can be 'neighbours',
    'excitation', 'inhibition','e-i balance', 'activity'
    neural_population (set of str): which neural populations to include.  For example, 'e_e' includes
        excitatory-to-excitatory connections, 'i_e' includes inhibitory-to-excitatory connections
    synapse_types (set of str): which synapse types to include.  'e' includes excitatory neurons, 'i' inhibitory ones
    open_output (bool): set to True to open the HTML file in the default browser
    show_buttons (bool): set to true to display the PyVis configuration UI in the HTML
    only_physics_buttons (bool): set to true to display the physics PyVis configuration UI in the HTML
    windows_size (tuple): the width and height of the window to display the graph in the HTML

    Returns:
    None

    """
    from pyvis import network as net
    import pathlib
    
    if neuron_types is None:
        neuron_types = { 'e', 'i' }
    if synapse_types is None:
        synapse_types = { 'e_e', 'i_e', 'e_i', 'i_i' }
    
    if isinstance( input, nx.Graph ):
        nx_graph = input
    elif isinstance( input, str ) and pathlib.Path( input ).suffix == '.graphml':
        nx_graph = nx.read_graphml( input )
    
    # make a pyvis network
    pyvis_graph = net.Network( notebook=notebook )
    pyvis_graph.width = f'{window_size[ 0 ]}px'
    pyvis_graph.height = f'{window_size[ 1 ]}px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in nx_graph.nodes( data=True ):
        # remove nodes that aren't of included type
        include = False
        for n in neuron_types:
            if node.split( '_' )[ 0 ] == n:
                include = True
                break
        if not include:
            continue
        
        pyvis_graph.add_node( node, **node_attrs )
    
    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in nx_graph.edges( data=True ):
        # remove edges that aren't of included type
        if (source.split( '_' )[ 0 ] in neuron_types and target.split( '_' )[ 0 ] in neuron_types) and (
                source.split( '_' )[ 0 ] + '_' + target.split( '_' )[ 0 ] in synapse_types):
            # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                # place at key 'value' the weight of the edge
                if 'e_' in source and 'e_' in target:
                    edge_attrs[ 'value' ] = edge_attrs[ 'weight' ]
                else:
                    edge_attrs[ 'value' ] = ''
            # add the edge
            pyvis_graph.add_edge( source, target, **edge_attrs, title='', arrows='to', dashes=True )
    
    def successors():
        successors = { }
        
        for n in pyvis_graph.get_nodes():
            successors[ n ] = set()
            for e in pyvis_graph.get_edges():
                if e[ 'from' ] == n:
                    successors[ n ].add( e[ 'to' ] )
        
        return successors
    
    def predecessors():
        predecessors = { }
        
        for n in pyvis_graph.get_nodes():
            predecessors[ n ] = set()
            for e in pyvis_graph.get_edges():
                if e[ 'to' ] == n:
                    predecessors[ n ].add( e[ 'from' ] )
        
        return predecessors
    
    def get_weights( i, js ):
        weights = [ ]
        
        for j in js:
            for e in pyvis_graph.get_edges():
                if e[ 'from' ] == j and e[ 'to' ] == i:
                    weights.append( e[ 'weight' ] )
        
        return weights
    
    # add neighbour data to node hover data
    pred = predecessors()
    succ = successors()
    type_colours = { 'excitatory': 'blue', 'inhibitory': 'red' }
    for node in pyvis_graph.nodes:
        node_pred = [ n for n in pred[ node[ 'id' ] ] ]
        node_pred_exc = [ n for n in pred[ node[ 'id' ] ] if 'e_' in n ]
        node_pred_inh = [ n for n in pred[ node[ 'id' ] ] if 'i_' in n ]
        node_succ = [ n for n in succ[ node[ 'id' ] ] ]
        node_succ_exc = [ n for n in succ[ node[ 'id' ] ] if 'e_' in n ]
        node_succ_inh = [ n for n in succ[ node[ 'id' ] ] if 'i_' in n ]
        
        total_excitation = np.sum( get_weights( node[ 'id' ], node_pred_exc ) )
        total_inhibition = np.sum( get_weights( node[ 'id' ], node_pred_inh ) )
        
        node[ 'title' ] += f'<center><h2>{node[ "id" ]}</h2></center>'
        node[ 'title' ] += f'<h3 style="color: {type_colours[ node[ "type" ] ]}">Kind: {node[ "type" ]}</h3>'
        node[ 'title' ] += f'<h3>E-I balance: {total_excitation - total_inhibition}</h3>'
        if 'e_' in node[ 'id' ]:
            node[ 'title' ] += f'<h3 style="color: {node[ "color" ]}">Attractor: {node[ "attractor" ]}</h3>'
        node[ 'title' ] += f'<h4>Excites excitatory ({len( node_succ_exc )}):</h4>' + ' '.join(
                [ f'<p style="color: {nx_graph.nodes[ n ][ "color" ]}">{n} ('
                  f'{nx_graph.nodes[ n ][ "attractor" ]})</p>' for n in node_succ_exc ]
                )
        node[ 'title' ] += f'<h4>Excites inhibitory ({len( node_succ_inh )}):</h4>' + ' '.join(
                [ f'<p style="color: red">{n}</p>' for n in node_succ_inh ]
                )
        node[ 'title' ] += f'<h4>Excited by ({len( node_pred_exc )}/{total_excitation}):</h4>' + ' '.join(
                [ f'<p style="color: {nx_graph.nodes[ n ][ "color" ]}">{n}</p>' for n in node_pred_exc ]
                )
        if 'e_' in node[ 'id' ]:
            node[ 'title' ] += f'<h4>Inhibited by ({len( node_pred_inh )}/{total_inhibition}):</h4>' + ' '.join(
                    [ f'<p style="color: red">{n}</p>' for n in node_pred_inh ]
                    )
        
        # TODO scale node value by activity and other metrics
        if scale_by == 'neighbours':
            node[ 'value' ] = len( node_pred + node_succ )
        elif scale_by == 'excitation':  # # more excitation, larger node
            node[ 'value' ] = total_excitation
        elif scale_by == 'inhibition':  # more inhibition, smaller node
            node[ 'value' ] = 1 - total_inhibition
        elif scale_by == 'e-i balance':
            node[ 'value' ] = total_excitation - total_inhibition
        elif scale_by == 'activity':
            pass
    
    for edge in pyvis_graph.edges:
        edge[ 'title' ] += f'<center><h3 style="color: ' \
                           f'{type_colours[ nx_graph.nodes[ edge[ "from" ] ][ "type" ] ]}">{edge[ "from" ]} → ' \
                           f'{edge[ "to" ]}</h3></center>'
        edge[ 'title' ] += f'<p style="color: {type_colours[ nx_graph.nodes[ edge[ "from" ] ][ "type" ] ]}">' \
                           f'Kind: {nx_graph.nodes[ edge[ "from" ] ][ "type" ]}</p>'
        edge[ 'title' ] += 'Weight: ' + str( edge[ 'weight' ] )
    
    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons( filter_=[ 'physics' ] )
        else:
            pyvis_graph.show_buttons()
    
    pyvis_graph.set_edge_smooth( 'dynamic' )
    pyvis_graph.toggle_hide_edges_on_drag( True )
    pyvis_graph.force_atlas_2based( damping=0.7 )
    
    # return and also save
    pyvis_graph.show( f'{output_filename}.html' )
    
    if open_output:
        import webbrowser
        import os
        
        webbrowser.open( f'file://{os.getcwd()}/{output_filename}.html' )


# TODO how much inhibition is each attractor giving?
# TODO how many inhibitory neurons are shared between attractors?
# TODO measure net excitation between attractors
# TODO measure connectedness within each attractor


def colour_by_attractor( g ):
    """Compute and store a different colour for each attractor in the NetworkX graph.
    
    Parameters:
    g (networkx.Graph): the graph for which to compute the attractor colour
    
    Returns:
    None
    
    """
    import cmasher as cmr
    
    num_attractors = len( set( nx.get_node_attributes( g, 'attractor' ).values() ) )
    colours = cmr.take_cmap_colors( 'tab20', num_attractors, return_fmt='int' )
    
    for n, v in g.nodes( data=True ):
        col = list( colours[ g.nodes[ n ][ 'attractor' ] ] )
        g.nodes[ n ][ 'color' ] = f'rgba({col[ 0 ]},{col[ 1 ]},{col[ 2 ]},1)'


def tag_weakly_connected_components( g ):
    """Compute the attractors in the NetworkX graph and store the information in the nodes by finding which
    weakly-connected component they belong to.
    A directed graph is weakly connected if and only if the graph is connected when the direction of the edge between
    nodes is ignored.  A connected graph is graph that is connected in the sense of a topological space, i.e.,
    there is a path from any point to any other point in the graph.
    
    Should correctly identify attractors for RCN when edges s.t. w(e) < w_max have previously been removed from the
    graph.
    
    Parameters:
    g (networkx.Graph): the graph for which to compute the attractors
    
    Returns:
    None
    
    """
    idx_components = { n: i for i, node_set in enumerate( nx.weakly_connected_components( g ) ) for n in node_set if
                       'e_' in n }
    for n, i in idx_components.items():
        g.nodes[ n ][ 'attractor' ] = i


def tag_attracting_components( g ):
    """Compute the attracting components in the NetworkX graph and store the information in the nodes.
    
    An attracting component in a directed graph G is a strongly connected component with the property that a random
    walker on the graph will never leave the component, once it enters the component.  The nodes in attracting
    components can also be thought of as recurrent nodes. If a random walker enters the attractor containing the node,
    then the node will be visited infinitely often.
    
    Parameters:
    g (networkx.Graph): the graph for which to compute the attracting components
    
    Returns:
    None
    
    """
    from networkx.algorithms.components import is_attracting_component
    
    attracting_map = { }
    for atr in set( nx.get_node_attributes( g, 'attractor' ).values() ):
        try:
            atr_nodes = [ n for n, v in g.nodes( data=True ) if v[ 'attractor' ] == atr ]
            atr_subgraph = g.subgraph( atr_nodes )
            is_attracting = is_attracting_component( atr_subgraph )
            for n in atr_nodes:
                attracting_map[ n ] = { 'is_attracting': is_attracting }
        except:
            continue
    
    nx.set_node_attributes( g, attracting_map )


def attractor_inhibition( input, normalise=False, output_filename='attractor_inhibition', comment='' ):
    """Compute the number of inhibitory connections incident to each attractor in the NetworkX graph.
    This should be useful to quantify the degree of inhibition of each attractor.
    For ex. attractor_inhibition_amount={0: 5, 1: 4} means that attractor 0 is receiving input from inhibitory cells
    via 5 connections, and attractor 1 via 4.
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    inhibition amount
    normalise (bool): if True the inhibition value is normalised by the total number of inhibitory connections
    output_filename (str): the name of the txt file to write results to in os.getcwd()
    
    Returns:
    attractor_inhibition_amount (dict): dictionary of attractors with the amount of inhibition as the value
    
    """
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
    import os
    from pprint import pprint
    
    if not isinstance( input, nx.Graph ) and isinstance( input, RecurrentCompetitiveNet ):
        g = rcn2nx( input )
    elif isinstance( input, nx.Graph ):
        g = input
    else:
        raise ValueError( 'input must be of type nx.Graph or RecurrentCompetitiveNet' )
    
    if normalise:
        norm = len( [ e for e in g.edges if 'i_' in e[ 0 ] ] )
    else:
        norm = 1
    
    attractor_inhibition_amount = { }
    
    for i in set( nx.get_node_attributes( g, 'attractor' ).values() ):
        attractor_nodes = [ n for n, v in g.nodes( data=True ) if 'e_' in n and v[ 'attractor' ] == i ]
        inhibitory_nodes = [ n for n in g.nodes if 'i_' in n ]
        subgraph = g.subgraph( attractor_nodes + inhibitory_nodes )
        
        attractor_inhibition_amount[ i ] = \
            len( [ e for e in subgraph.edges if e[ 0 ] in inhibitory_nodes and e[ 1 ] in attractor_nodes ] ) / norm
    
    if not os.path.exists( f'{os.getcwd()}/{output_filename}.txt' ):
        with open( f'{os.getcwd()}/{output_filename}.txt', 'w' ) as f:
            f.write( """Generated via the graphing/attractor_inhibition function.
			
	Compute the number of inhibitory connections incident to each attractor in the NetworkX graph.
	This should be useful to quantify the degree of inhibition of each attractor.
	For ex. attractor_inhibition_amount={0: 5, 1: 4} means that attractor 0 is receiving input from inhibitory cells
	via
	5 connections, and attractor 1 via 4.
	
	Results:
	
	""" )
            f.write( f'\t{comment}\n\t' )
            pprint( attractor_inhibition_amount, stream=f )
    else:
        with open( f'{os.getcwd()}/{output_filename}.txt', 'a' ) as f:
            f.write( f'\t{comment}\n\t' )
            pprint( attractor_inhibition_amount, stream=f )
    
    return attractor_inhibition_amount


def attractor_connectivity( input, output_filename='attractor_connectivity', comment='' ):
    """Computes the average node connectivity within each attractor in the NetworkX graph.
    This should be useful to quantify the amount of self-excitation each attractor has.
    The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G.
    Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
    along with their incident edges) to disconnect them.
    This functions is likely to be slow on the full network ( ~4 minutes on Apple M1).
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    connectivity
    output_filename (str): the name of the txt file to write results to in os.getcwd()
    
    Returns:
    attractor_connectivity_amount (dict): dictionary of attractors with the average node connectivity as the value
    
"""
    from networkx.algorithms.connectivity.connectivity import average_node_connectivity
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
    import os
    from pprint import pprint
    
    if not isinstance( input, nx.Graph ) and isinstance( input, RecurrentCompetitiveNet ):
        g = rcn2nx( input )
    elif isinstance( input, nx.Graph ):
        g = input
    else:
        raise ValueError( 'input must be of type nx.Graph or RecurrentCompetitiveNet' )
    
    attractor_connectivity_amount = { }
    
    for i in set( nx.get_node_attributes( g, 'attractor' ).values() ):
        attractor_nodes = [ n for n, v in g.nodes( data=True ) if 'e_' in n and v[ 'attractor' ] == i ]
        subgraph = g.subgraph( attractor_nodes )
        
        attractor_connectivity_amount[ i ] = average_node_connectivity( subgraph )
    
    if not os.path.exists( f'{os.getcwd()}/{output_filename}.txt' ):
        with open( f'{os.getcwd()}/{output_filename}.txt', 'w' ) as f:
            f.write( """Generated via the graphing/attractor_connectivity function.
			
	Computes the average node connectivity within each attractor in the NetworkX graph.
	This should be useful to quantify the amount of self-excitation each attractor has.
	The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G.
	Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
	along with their incident edges) to disconnect them.
	
	Results:

	""" )
            f.write( f'\t{comment}\n' )
            pprint( attractor_connectivity_amount, stream=f )
    
    else:
        with open( f'{os.getcwd()}/{output_filename}.txt', 'a' ) as f:
            f.write( f'\t{comment}\n' )
            pprint( attractor_connectivity_amount, stream=f )
    
    return attractor_connectivity_amount


files = [
        'initial.html', 'initial.graphml', 'initial_complete.graphml',
        'first.html', 'first.graphml', 'first_complete.graphml',
        'second.html', 'second.graphml', 'second_complete.graphml',
        'rcn_population_spiking.png',
        'attractor_inhibition.txt', 'attractor_connectivity.txt',
        ]


def save_graph_results( folder='interesting_graph_results', additional_files=None, comments='' ):
    """Saves files resulting from graphing in a folder

    Parameters:
    folder (str): the base folder under which results will be saved
    additional_files (list): any additional files to move along with the default ones
    comments (str): any comments to add to the folder.  Will be written into `comments.txt`

    Returns:
    None

"""
    import shutil
    import datetime
    import os
    
    if additional_files is None:
        additional_files = [ ]
    
    new_folder = './' + folder + '/' + str( datetime.datetime.now().date() ) + '_' + \
                 str( datetime.datetime.now().time().replace( microsecond=0 ) ).replace( ':', '.' )
    os.makedirs( new_folder )
    
    files_local = files + additional_files
    
    count = 0
    for f in files_local:
        try:
            shutil.move( os.getcwd() + '/' + f, new_folder )
            count += 1
        except:
            continue
    
    if count > 0:
        print( f'Moved {count} files to {new_folder}' )
        
        if comments:
            with open( f'{new_folder}/comments.txt', 'w' ) as f:
                f.write( comments )
        else:
            with open( f'{new_folder}/comments.txt', 'w' ) as f:
                f.write( 'Nothing particular to remark.' )
    
    else:
        os.rmdir( new_folder )
        print( 'No files to move' )


def clean_folder( additional_files=None ):
    import os
    
    if additional_files is None:
        additional_files = [ ]
    
    files_local = files + additional_files
    
    count = 0
    for f in files_local:
        try:
            os.remove( f )
            count += 1
        except:
            continue
    
    print( f'Removed {count} files' )
