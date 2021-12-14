# import
import os

import numpy as np
import pandas as pd

from brian2 import second, prefs, NeuronGroup, Synapses

from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms


def tag_weakly_connected_components( g ):
    import networkx as nx
    
    idx_components = { u: i for i, node_set in enumerate( nx.weakly_connected_components( g ) ) for u in node_set }
    for n, i in idx_components.items():
        g.nodes[ n ][ 'attractor' ] = i


def tag_attracting_components( g ):
    import networkx as nx
    from networkx.algorithms.components import is_attracting_component
    
    attracting_map = { }
    for atr in set( nx.get_node_attributes( g, 'attractor' ).values() ):
        atr_nodes = [ n for n, v in g.nodes( data=True ) if v[ 'attractor' ] == atr ]
        atr_subgraph = g.subgraph( atr_nodes )
        is_attracting = is_attracting_component( atr_subgraph )
        for n in atr_nodes:
            attracting_map[ n ] = { 'is_attracting': is_attracting }
    
    nx.set_node_attributes( g, attracting_map )


def colour_by_attractor( g ):
    import networkx as nx
    import cmasher as cmr
    
    num_attractors = len( set( nx.get_node_attributes( g, 'attractor' ).values() ) )
    colours = cmr.take_cmap_colors( 'tab20', num_attractors, return_fmt='hex' )
    
    for n, v in g.nodes( data=True ):
        g.nodes[ n ][ 'color' ] = colours[ g.nodes[ n ][ 'attractor' ] ]


def draw_graph3( networkx_graph, notebook=False, output_filename='graph',
                 scale_by='',
                 open_output=False,
                 show_buttons=False, only_physics_buttons=False ):
    from pyvis import network as net
    
    # make a pyvis network
    pyvis_graph = net.Network( notebook=notebook )
    pyvis_graph.width = '1000px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in networkx_graph.nodes( data=True ):
        pyvis_graph.add_node( node, **node_attrs )
    
    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges( data=True ):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs[ 'value' ] = edge_attrs[ 'weight' ]
        # add the edge
        pyvis_graph.add_edge( source, target, **edge_attrs, title='', arrows='to', dashes=True )
    
    # add neighbor data to node hover data
    neighbour_map = pyvis_graph.get_adj_list()
    for node in pyvis_graph.nodes:
        node[ 'title' ] += f' Node: {node[ "id" ]}<br>' + ' Excites:<br>' + '<br>'.join(
                node[ 'excites' ] ) + '<br> Inhibited by:<br>' + '<br>'.join( node[ 'inhibitors' ] )
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


# Build NetworkX graph
def rcn2nx( net, e_neurons, i_neurons, remove_zero_weight_edges=True, output_filename='graph',
            colour_attractors=True ):
    import networkx as nx
    
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
    g.add_nodes_from( e_nodes, color='blue', title='', type='excitatory' )
    for i, j, w in zip( e2e_edges_pre, e2e_edges_post, e2e_edge_weights ):
        g.add_edge( f'e_{i}', f'e_{j}', weight=w )
    
    # Classify excitatory neurons
    tag_weakly_connected_components( g )
    tag_attracting_components( g )
    if colour_attractors:
        colour_by_attractor( g )
    
    # Add inhibitory neurons
    i_nodes = [ f'i_{i}' for i in i_neurons ]
    g.add_nodes_from( i_nodes, color='red', title='', type='inhibitory' )
    for i, j, w in zip( i2e_edges_pre, i2e_edges_post, i2e_edge_weights ):
        g.add_edge( f'i_{i}', f'e_{j}', weight=w )
    
    # compute neighbourhoods
    for n in e_nodes:
        g.nodes[ n ][ 'inhibitors' ] = [ i for i in list( g.predecessors( n ) ) if 'i_' in i ]
        g.nodes[ n ][ 'excites' ] = list( g.successors( n ) )
    for n in i_nodes:
        g.nodes[ n ][ 'inhibitors' ] = [ ]
        g.nodes[ n ][ 'excites' ] = list( g.successors( n ) )
    
    # if output_filename:
    #     nx.write_graphml( g, f'{os.getcwd()}/{output_filename}.graphml' )
    
    return g


prefs.codegen.target = 'numpy'

# Helper modules
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from plotting_functions.plot import *

# 1 ------ initializing/running network ------

make_plots = True
plasticity_rule = 'LR3'
parameter_set = '2.0'
# plasticity_rule = 'LR4'
# parameter_set = '2.0'

rcn = RecurrentCompetitiveNet(
        plasticity_rule=plasticity_rule,
        parameter_set=parameter_set,
        t_run=3 * second,
        save_path=os.getcwd() )

rcn.stimulus_pulse = True

rcn.net_init()

# select neurons from both attractors
num_subsample = 15
assert num_subsample % 3 == 0
# np.random.seed( 0 )
E_neuron_subgroup = np.concatenate(
        [ np.random.choice( 64, int( num_subsample / 3 ), replace=False ),
          np.random.choice( range( 100, 164 ), int( num_subsample / 3 ), replace=False ),
          np.random.choice( range( 164, 256 ), int( num_subsample / 3 ), replace=False ) ]
        )
I_neuron_subgroup = np.random.choice( rcn.N_input_i, num_subsample, replace=False )

g = rcn2nx( rcn, E_neuron_subgroup, I_neuron_subgroup, output_filename='initial' )
draw_graph3( g, output_filename='initial', open_output=True, show_buttons=True,
             only_physics_buttons=True )

# -------- First attractor

rcn.set_E_E_plastic( plastic=True )

rcn.set_stimulus_e( stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=0 )
rcn.set_stimulus_i( stimulus='flat_to_I', frequency=rcn.stim_freq_i )

rcn.run_net( period=2 )

g = rcn2nx( rcn, E_neuron_subgroup, I_neuron_subgroup, output_filename='first' )
draw_graph3( g, output_filename='first', open_output=True, show_buttons=True,
             only_physics_buttons=True )

# --------- Second attractor

# TODO set edges based on weight
# TODO plot all neurons as clusters


rcn.stimulus_pulse_duration = 5 * second

rcn.set_stimulus_e( stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=100 )
rcn.set_stimulus_i( stimulus='flat_to_I', frequency=rcn.stim_freq_i )

rcn.run_net( period=2 )

g = rcn2nx( rcn, E_neuron_subgroup, I_neuron_subgroup, output_filename='second' )
draw_graph3( g, output_filename='second', open_output=True, show_buttons=True,
             only_physics_buttons=True )

plot_rcn_spiketrains_histograms(
        Einp_spks=rcn.get_Einp_spks()[ 0 ],
        Einp_ids=rcn.get_Einp_spks()[ 1 ],
        stim_E_size=rcn.stim_size_e,
        E_pop_size=rcn.N_input_e,
        Iinp_spks=rcn.get_Iinp_spks()[ 0 ],
        Iinp_ids=rcn.get_Iinp_spks()[ 1 ],
        stim_I_size=rcn.stim_size_i,
        I_pop_size=rcn.N_input_i,
        E_spks=rcn.get_E_spks()[ 0 ],
        E_ids=rcn.get_E_spks()[ 1 ],
        I_spks=rcn.get_I_spks()[ 0 ],
        I_ids=rcn.get_I_spks()[ 1 ],
        t_run=6,
        path_to_plot=os.getcwd(),
        show=True )

import networkx as nx
#
# e_nodes = [ n for n, v in g.nodes( data=True ) if v[ 'type' ] == 'excitatory' ]
# e_subgraph = g.subgraph( e_nodes )
#
# tag_weakly_connected_components( e_subgraph )
# tag_attracting_components( e_subgraph )
# colour_by_attractor( e_subgraph )
#
# draw_graph3( e_subgraph, output_filename='third', open_output=True, show_buttons=True )
