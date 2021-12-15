import networkx as nx
import numpy as np


def tag_weakly_connected_components( g ):
    idx_components = { n: i for i, node_set in enumerate( nx.weakly_connected_components( g ) ) for n in node_set if
                       'e_' in n }
    for n, i in idx_components.items():
        g.nodes[ n ][ 'attractor' ] = i


def tag_attracting_components( g ):
    from networkx.algorithms.components import is_attracting_component
    
    attracting_map = { }
    for atr in set( nx.get_node_attributes( g, 'attractor' ).values() ):
        atr_nodes = [ n for n, v in g.nodes( data=True ) if v[ 'attractor' ] == atr ]
        atr_subgraph = g.subgraph( atr_nodes )
        is_attracting = is_attracting_component( atr_subgraph )
        for n in atr_nodes:
            attracting_map[ n ] = { 'is_attracting': is_attracting }
    
    nx.set_node_attributes( g, attracting_map )


def rgb_to_hex( r, g, b ):
    return ('#{:02X}{:02X}{:02X}').format( r, g, b )


def hex_to_rgb( h ):
    h = h.lstrip( '#' )
    
    return tuple( int( h[ i:i + 2 ], 16 ) for i in (0, 2, 4) )


def colour_by_attractor( g ):
    import cmasher as cmr
    
    num_attractors = len( set( nx.get_node_attributes( g, 'attractor' ).values() ) )
    colours = cmr.take_cmap_colors( 'tab20', num_attractors, return_fmt='int' )
    
    for n, v in g.nodes( data=True ):
        col = list( colours[ g.nodes[ n ][ 'attractor' ] ] )
        g.nodes[ n ][ 'color' ] = f'rgba({col[ 0 ]},{col[ 1 ]},{col[ 2 ]},1)'


# TODO remove edges smaller than max_weights?
# TODO hide inhibitory?
# Build NetworkX graph
def rcn2nx( net, neurons_subsample=None, subsample_attractors=False, seed=None,
            remove_zero_weight_edges=True, output_filename='graph',
            colour_attractors=True ):
    import os
    
    # ------ Subsample graph nodes
    if neurons_subsample:
        e_neurons_subsample = np.clip( neurons_subsample[ 0 ], 0, len( net.E ) )
        i_neurons_subsample = np.clip( neurons_subsample[ 1 ], 0, len( net.I ) )
        
        if seed:
            np.random.seed( seed )
        if subsample_attractors:
            assert e_neurons_subsample % 3 == 0
            e_neurons = np.concatenate(
                    [ np.random.choice( range( 0, 64 ), int( e_neurons_subsample / 3 ), replace=False ),
                      np.random.choice( range( 100, 164 ), int( e_neurons_subsample / 3 ), replace=False ),
                      np.random.choice( range( 164, 256 ), int( e_neurons_subsample / 3 ), replace=False ) ]
                    )
            i_neurons = np.random.choice( range( 0, len( net.I ) ), i_neurons_subsample, replace=False )
        else:
            e_neurons = np.random.choice( range( 0, len( net.E ) ), e_neurons_subsample, replace=False )
            i_neurons = np.random.choice( range( 0, len( net.I ) ), i_neurons_subsample, replace=False )
    else:
        e_neurons = np.array( range( 0, len( net.E ) ) )
        i_neurons = np.array( range( 0, len( net.I ) ) )
    
    # ------- workaround for bug in Brian2 (not needed?)
    # import brian2._version
    # from packaging import version
    #
    # brian_version = brian2._version.get_versions()[ 'version' ]
    # if version.parse( brian_version ) > version.parse( '2.5' ):
    #     syn_indices = np.arange( len( net.I_E ) )[ np.isin( net.I_E.i, i_neurons ) & np.isin( net.I_E.j, e_neurons ) ]
    #
    #     syn_indices_2 = [ (i, j) for i, j in zip( net.I_E.i, net.I_E.j ) if i in i_neurons and j in e_neurons ]
    # else:
    e2e_edges_pre = net.E_E.i[ e_neurons, e_neurons ].tolist()
    e2e_edges_post = net.E_E.j[ e_neurons, e_neurons ].tolist()
    e2e_edge_weights = net.E_E.w_[ e_neurons, e_neurons ].tolist()
    i2e_edges_pre = net.I_E.i[ i_neurons, e_neurons ].tolist()
    i2e_edges_post = net.I_E.j[ i_neurons, e_neurons ].tolist()
    i2e_edge_weights = net.I_E.w_[ i_neurons, e_neurons ].tolist()
    
    if remove_zero_weight_edges:
        e2e_edges_pre = [ k for i, k in enumerate( e2e_edges_pre ) if e2e_edge_weights[ i ] != 0 ]
        e2e_edges_post = [ k for i, k in enumerate( e2e_edges_post ) if e2e_edge_weights[ i ] != 0 ]
        e2e_edge_weights = [ k for i, k in enumerate( e2e_edge_weights ) if e2e_edge_weights[ i ] != 0 ]
    
    g = nx.DiGraph()
    
    # Add excitatory neurons
    e_nodes = [ f'e_{i}' for i in e_neurons ]
    for n in e_nodes:
        g.add_node( n, label=n, color='rgba(0,0,255,1)', title='', type='excitatory' )
    for i, j, w in zip( e2e_edges_pre, e2e_edges_post, e2e_edge_weights ):
        g.add_edge( f'e_{i}', f'e_{j}', weight=w )
    
    # Classify excitatory neurons
    tag_weakly_connected_components( g )
    tag_attracting_components( g )
    if colour_attractors:
        colour_by_attractor( g )
    
    # Add inhibitory neurons
    i_nodes = [ f'i_{i}' for i in i_neurons ]
    for n in i_nodes:
        g.add_node( n, label=n, color='rgba(255,0,0,0.5)', title='', type='inhibitory' )
    for i, j, w in zip( i2e_edges_pre, i2e_edges_post, i2e_edge_weights ):
        g.add_edge( f'i_{i}', f'e_{j}', weight=w )
    
    #  GraphML does not support list or any non-primitive type, so we save before computing them
    if output_filename:
        nx.write_graphml( g, f'{os.getcwd()}/{output_filename}.graphml' )
    
    # compute neighbourhoods
    # for n in e_nodes:
    #     g.nodes[ n ][ 'inhibitors' ] = [ i for i in list( g.predecessors( n ) ) if 'i_' in i ]
    #     g.nodes[ n ][ 'excites' ] = list( g.successors( n ) )
    # for n in i_nodes:
    #     g.nodes[ n ][ 'inhibitors' ] = [ ]
    #     g.nodes[ n ][ 'excites' ] = list( g.successors( n ) )
    
    return g


# TODO fade non-neighbourhood on click
# TODO add attracting_components info to viz
def nx2pyvis( networkx_graph, notebook=False, output_filename='graph',
              scale_by='',
              open_output=False,
              show_buttons=False, only_physics_buttons=False ):
    from pyvis import network as net
    
    # make a pyvis network
    pyvis_graph = net.Network( notebook=notebook )
    pyvis_graph.width = '1000px'
    pyvis_graph.height = '1000px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in networkx_graph.nodes( data=True ):
        pyvis_graph.add_node( node, **node_attrs )
    
    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges( data=True ):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            if 'i_' not in source and 'i_' not in target:
                edge_attrs[ 'value' ] = edge_attrs[ 'weight' ]
            else:
                edge_attrs[ 'value' ] = ''
        # add the edge
        pyvis_graph.add_edge( source, target, **edge_attrs, title='', arrows='to', dashes=True )
    
    # add neighbor data to node hover data
    neighbour_map = pyvis_graph.get_adj_list()
    type_colours = { 'excitatory': 'blue', 'inhibitory': 'red' }
    for node in pyvis_graph.nodes:
        node[ 'title' ] += f'<center><h2>{node[ "id" ]}</h2></center>'
        node[ 'title' ] += f'<h3 style="color: {type_colours[ node[ "type" ] ]}">Kind: {node[ "type" ]}</h3>'
        if 'e_' in node[ 'id' ]:
            node[ 'title' ] += f'<h3 style="color: {node[ "color" ]}">Attractor: {node[ "attractor" ]}</h3>'
        node[ 'title' ] += '<h4>Excites:</h4>' + ' '.join(
                [ f'<p style="color: {networkx_graph.nodes[ n ][ "color" ]}">{n} ('
                  f'{networkx_graph.nodes[ n ][ "attractor" ]})</p>'
                  for n in neighbour_map[ node[ 'id' ] ] if 'e_' in n ]
                )
        if 'e_' in node[ 'id' ]:
            node[ 'title' ] += '<h4>Inhibited by:</h4>' + ' '.join(
                    [ f'<p style="color: red">{n}</p>' for n in neighbour_map[ node[ 'id' ] ] if 'i_' in n ]
                    )
        
        # TODO scale node value by activity or other metrics
        if scale_by == 'neighbours':
            node[ 'value' ] = len( neighbour_map[ node[ 'id' ] ] )
        elif scale_by == 'activity':
            pass
        elif scale_by == 'inhibition':
            pass
    
    for edge in pyvis_graph.edges:
        edge[ 'title' ] = ' Weight:<br>' + str( edge[ 'weight' ] )
    
    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons( filter_=[ 'physics' ] )
        else:
            pyvis_graph.show_buttons()
    
    pyvis_graph.set_edge_smooth( 'dynamic' )
    pyvis_graph.toggle_hide_edges_on_drag( True )
    pyvis_graph.force_atlas_2based()
    
    # return and also save
    pyvis_graph.show( f'{output_filename}.html' )
    
    if open_output:
        import webbrowser
        import os
        
        webbrowser.open( f'file://{os.getcwd()}/{output_filename}.html' )
