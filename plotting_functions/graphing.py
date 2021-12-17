import networkx as nx
import numpy as np


def rgb_to_hex( r, g, b ):
	return ('#{:02X}{:02X}{:02X}').format( r, g, b )


def hex_to_rgb( h ):
	h = h.lstrip( '#' )
	
	return tuple( int( h[ i:i + 2 ], 16 ) for i in (0, 2, 4) )


def rcn2nx( net, neurons_subsample=None, subsample_attractors=False, seed=None,
            remove_edges_threshold=0, output_filename='graph' ):
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
	
	# ------- Make sure that there are no duplicates in neurons subsample or Brian2 indexing runs into a bug
	assert len( np.unique( e_neurons ) ) == len( e_neurons ) or len( np.unique( i_neurons ) ) == len( i_neurons )
	e2e_edges_pre = net.E_E.i[ e_neurons, e_neurons ].tolist()
	e2e_edges_post = net.E_E.j[ e_neurons, e_neurons ].tolist()
	e2e_edge_weights = net.E_E.w_[ e_neurons, e_neurons ].tolist()
	i2e_edges_pre = net.I_E.i[ i_neurons, e_neurons ].tolist()
	i2e_edges_post = net.I_E.j[ i_neurons, e_neurons ].tolist()
	i2e_edge_weights = net.I_E.w_[ i_neurons, e_neurons ].tolist()
	
	if remove_edges_threshold is not None:
		e2e_edges_pre = [ k for i, k in enumerate( e2e_edges_pre ) if
		                  e2e_edge_weights[ i ] > remove_edges_threshold ]
		e2e_edges_post = [ k for i, k in enumerate( e2e_edges_post ) if
		                   e2e_edge_weights[ i ] > remove_edges_threshold ]
		e2e_edge_weights = [ k for i, k in enumerate( e2e_edge_weights ) if
		                     e2e_edge_weights[ i ] > remove_edges_threshold ]
	
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
	
	#  GraphML does not support list or any non-primitive type
	if output_filename:
		nx.write_graphml( g, f'{os.getcwd()}/{output_filename}.graphml' )
	
	return g


# TODO fade non-neighbourhood on click
# TODO add attracting_components info to viz
# TODO hide inhibitory?
def nx2pyvis( input, notebook=False, output_filename='graph',
              scale_by='',
              neural_populations=None,
              open_output=False, show_buttons=False, only_physics_buttons=False, window_size=(1000, 1000) ):
	from pyvis import network as net
	
	import pathlib
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
		pyvis_graph.add_node( node, **node_attrs )
	
	# for each edge and its attributes in the networkx graph
	for source, target, edge_attrs in nx_graph.edges( data=True ):
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
				[ f'<p style="color: {nx_graph.nodes[ n ][ "color" ]}">{n} ('
				  f'{nx_graph.nodes[ n ][ "attractor" ]})</p>'
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
# TODO measure net exhitation between attractors
# TODO measure connectedness within each attractor

"""Colour each attractor in a different colour."""


def colour_by_attractor( g ):
	import cmasher as cmr
	
	num_attractors = len( set( nx.get_node_attributes( g, 'attractor' ).values() ) )
	colours = cmr.take_cmap_colors( 'tab20', num_attractors, return_fmt='int' )
	
	for n, v in g.nodes( data=True ):
		col = list( colours[ g.nodes[ n ][ 'attractor' ] ] )
		g.nodes[ n ][ 'color' ] = f'rgba({col[ 0 ]},{col[ 1 ]},{col[ 2 ]},1)'


"""Compute the attractors and store the information in the nodes by finding which weakly-connected component they belong
to.
Should work correctly for RCN when edges s.t. w(e) < w_max are removed from the graph."""


def tag_weakly_connected_components( g ):
	idx_components = { n: i for i, node_set in enumerate( nx.weakly_connected_components( g ) ) for n in node_set if
	                   'e_' in n }
	for n, i in idx_components.items():
		g.nodes[ n ][ 'attractor' ] = i


def tag_attracting_components( g ):
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


"""Compute what fraction of inhibitory cells connected to each attractor and the number of these connections to
quantify the degree of inhibition of each attractor.
For ex. attractor_inhibition_amount={0: 5, 1: 4} means that attractor 0 is receiving input from inhibitory cells via
5 connections, and attractor 1 via 4.
"""


def attractor_inhibition( input ):
	from networkx.algorithms.centrality import group_in_degree_centrality
	from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
	
	if not isinstance( input, nx.Graph ) and isinstance( input, RecurrentCompetitiveNet ):
		input = rcn2nx( input )
	
	attractor_inhibition_amount = { }
	
	for i in set( nx.get_node_attributes( input, 'attractor' ).values() ):
		attractor_nodes = [ n for n, v in input.nodes( data=True ) if 'e_' in n and v[ 'attractor' ] == i ]
		inhibitory_nodes = [ n for n in input.nodes if 'i_' in n ]
		subgraph = input.subgraph( attractor_nodes + inhibitory_nodes )
		
		attractor_inhibition_amount[ i ] = \
			len( [ e for e in input.edges if e[ 0 ] in inhibitory_nodes and e[ 1 ] in attractor_nodes ] )
	
	return attractor_inhibition_amount


"""Computes the average connectivity within each attractor.
The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of
G.  Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
along with their incident edges) to disconnect them.
"""


def attractor_connectivity( input ):
	from networkx.algorithms.connectivity.connectivity import average_node_connectivity
	from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
	
	if not isinstance( input, nx.Graph ) and isinstance( input, RecurrentCompetitiveNet ):
		input = rcn2nx( input )
	
	attractor_connectivity_amount = { }
	
	for i in set( nx.get_node_attributes( input, 'attractor' ).values() ):
		attractor_nodes = [ n for n, v in input.nodes( data=True ) if 'e_' in n and v[ 'attractor' ] == i ]
		subgraph = input.subgraph( attractor_nodes )
		
		attractor_connectivity_amount[ i ] = average_node_connectivity( subgraph )
	
	return attractor_connectivity_amount


def save_graph_results( comment='', folder='interesting_graph_results', additional_files=None ):
	import shutil
	import datetime
	import os
	
	if additional_files is None:
		additional_files = [ ]
	
	new_folder = './' + folder + '/' + str( datetime.datetime.now().date() ) + '_' + \
	             str( datetime.datetime.now().time().replace( microsecond=0 ) ).replace( ':', '.' )
	os.makedirs( new_folder )
	
	files = [
			        'initial.html', 'initial.graphml', 'initial_complete.graphml',
			        'first.html', 'first.graphml', 'first_complete.graphml',
			        'second.html', 'second.graphml', 'second_complete.graphml',
			        'rcn_population_spiking.png',
			        'test.graphml'
			        ] + additional_files
	
	count = 0
	for f in files:
		try:
			shutil.move( os.getcwd() + '/' + f, new_folder )
			count += 1
		except:
			continue
	
	if count > 0:
		print( f'Moved {count} files to {new_folder}' )
	else:
		os.rmdir( new_folder )
		print( 'No files to move' )
