from plotting_functions.graphing import *

g = nx.read_graphml( 'interesting_graph_results/2021-12-30_13.03.26/second.graphml' )
nx2pyvis( g, output_filename='test' )

print( 'Nodes:', len( g.nodes ) )
print( 'Edges:', len( g.edges ) )
print( attractor_inhibition( g ) )
print( attractor_connectivity( g ) )
