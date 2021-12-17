import networkx as nx
import numpy as np


def rgb_to_hex( r, g, b ):
	return ('#{:02X}{:02X}{:02X}').format( r, g, b )


def hex_to_rgb( h ):
	h = h.lstrip( '#' )
	
	return tuple( int( h[ i:i + 2 ], 16 ) for i in (0, 2, 4) )


# TODO include all connections
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
	e2e_edges_pre = rcn.E_E.i[ e_neurons, e_neurons ].tolist()
	e2e_edges_post = rcn.E_E.j[ e_neurons, e_neurons ].tolist()
	e2e_edge_weights = rcn.E_E.w_[ e_neurons, e_neurons ].tolist()
	i2e_edges_pre = rcn.I_E.i[ i_neurons, e_neurons ].tolist()
	i2e_edges_post = rcn.I_E.j[ i_neurons, e_neurons ].tolist()
	i2e_edge_weights = rcn.I_E.w_[ i_neurons, e_neurons ].tolist()
	
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
	
	# GraphML does not support list or any non-primitive type
	if output_filename:
		nx.write_graphml( g, f'{os.getcwd()}/{output_filename}.graphml' )
	
	return g


# TODO fade non-neighbourhood on click
# TODO add attracting_components info to viz
# TODO hide inhibitory?
def nx2pyvis( input, notebook=False, output_filename='graph',
              scale_by='',
              neural_populations=None,
              synapse_types=None,
              open_output=True, show_buttons=True, only_physics_buttons=True, window_size=(1000, 1000) ):
	"""Given an instance of a Networx.Graph builds the corresponding PyVis graph for display purposes.

	Parameters:
	input (nx.Graph or str): the NetworkX graph to convert to PyVis or the path to a GraphML file to convert
	output_filename (str): the name of the file HTML to save to disk in os.getcwd()
	TODO - scale_by (str): the node parameter whose value is used to visually scale the nodes
	TODO - neural_population (): which neural populations to include
	TODO - synapse_types (): which synapse types to include
	open_output (bool): set to True to open the HTML file in the default browser
	show_buttons (bool): set to true to display the PyVis configuration UI in the HTML
	only_physics_buttons (bool): set to true to display the physics PyVis configuration UI in the HTML
	windows_size (tuple): the width and height of the window to display the graph in the HTML

	Returns:
	None

	"""
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


def attractor_inhibition( input, normalise=False, output_filename='attractor_inhibition' ):
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
	
	with open( f'{os.getcwd()}/{output_filename}.txt', 'w' ) as f:
		f.write( """Generated via the graphing/attractor_inhibition function.
		
Compute the number of inhibitory connections incident to each attractor in the NetworkX graph.
This should be useful to quantify the degree of inhibition of each attractor.
For ex. attractor_inhibition_amount={0: 5, 1: 4} means that attractor 0 is receiving input from inhibitory cells via
5 connections, and attractor 1 via 4.

Result:
""" )
		pprint( attractor_inhibition_amount, stream=f )
	
	return attractor_inhibition_amount


def attractor_connectivity( input, output_filename='attractor_connectivity' ):
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
	
	for i in set( nx.get_node_attributes( input, 'attractor' ).values() ):
		attractor_nodes = [ n for n, v in input.nodes( data=True ) if 'e_' in n and v[ 'attractor' ] == i ]
		subgraph = input.subgraph( attractor_nodes )
		
		attractor_connectivity_amount[ i ] = average_node_connectivity( subgraph )
	
	with open( f'{os.getcwd()}/{output_filename}.txt', 'w' ) as f:
		f.write( """Generated via the graphing/attractor_connectivity function.
		
Computes the average node connectivity within each attractor in the NetworkX graph.
This should be useful to quantify the amount of self-excitation each attractor has.
The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G.
Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
along with their incident edges) to disconnect them.

Result:
""" )
		pprint( attractor_connectivity_amount, stream=f )
	
	return attractor_connectivity_amount


# TODO save generated statistics too
def save_graph_results( folder='interesting_graph_results', additional_files=None, comments='' ):
	"""Saves files resulting from graphing in a folder

	Parameters:
	folder (str): the base folder under which results will be saved
	additional_files (list): any additional files to move along with the default ones
	TODO - comment (str): any comments to add to the folder

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
	
	files = [
			        'initial.html', 'initial.graphml', 'initial_complete.graphml',
			        'first.html', 'first.graphml', 'first_complete.graphml',
			        'second.html', 'second.graphml', 'second_complete.graphml',
			        'rcn_population_spiking.png',
			        'attractor_inhibition.txt', 'attractor_connectivity.txt',
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
		
		if comments:
			with open( f'{new_folder}/comments.txt', 'w' ) as f:
				f.write( comments )
		else:
			with open( f'{new_folder}/comments.txt', 'w' ) as f:
				f.write( 'Nothing particular to remark.' )
	
	else:
		os.rmdir( new_folder )
		print( 'No files to move' )
