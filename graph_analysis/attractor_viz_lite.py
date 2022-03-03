from plotting_functions.graphing import *

complete = True
folder = '2022-01-07_17.05.18'
graph_filename = f'interesting_graph_results/{folder}/second.graphml' if not complete else \
    f'interesting_graph_results/{folder}/second_complete.graphml'

g = nx.read_graphml( graph_filename )
nx2pyvis( g, output_filename='test', open_output=False )

# print( 'Nodes:', len( g.nodes ) )
# print( 'Edges:', len( g.edges ) )
# print( 'Inhibition', attractor_statistics( g, 'inhibition' ) )
# print( 'Excitation', attractor_statistics( g, 'excitation' ) )
# print( 'Self-excitation', attractor_statistics( g, 'self-excitation' ) )
# print( 'Algebraic connectivity', attractor_algebraic_connectivity( g ) )
# print( 'Connectivity', attractor_connectivity( g ) )
# print( 'Mutual inhibition', attractor_mutual_inhibition( g ) )
print( harcoded_attractor_algebraic_connectivity( g ) )
