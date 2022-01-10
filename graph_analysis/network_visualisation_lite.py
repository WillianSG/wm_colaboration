from plotting_functions.graphing import *

g = nx.read_graphml( 'interesting_graph_results/2022-01-07_17.05.18/second.graphml' )
nx2pyvis( g, output_filename='test', open_output=False )

print( 'Nodes:', len( g.nodes ) )
print( 'Edges:', len( g.edges ) )
print( 'Inhibition', attractor_statistics( g, 'inhibition' ) )
print( 'Excitation', attractor_statistics( g, 'excitation' ) )
print( 'Self-excitation', attractor_statistics( g, 'self-excitation' ) )
print( 'Connectivity', attractor_connectivity( g ) )
print( 'Mutual inhibition', attractor_mutual_inhibition( g ) )
