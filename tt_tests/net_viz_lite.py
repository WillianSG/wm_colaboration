import networkx as nx

from plotting_functions.graphing import *

g = nx.read_graphml( 'interesting_graph_results/1/second.graphml' )
nx2pyvis( g, output_filename='test', open_output=True, show_buttons=True, only_physics_buttons=True )

print( attractor_inhibition( g ) )
print( attractor_connectivity( g ) )
