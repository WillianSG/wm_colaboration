from plotting_functions.graphing import *

g = nx.read_graphml( 'interesting_graph_results/2021-12-17_10.55.47/second.graphml' )
nx2pyvis( g, output_filename='test' )

print('Nodes:',len(g))
print( attractor_inhibition( g ) )
print( attractor_connectivity( g ) )
