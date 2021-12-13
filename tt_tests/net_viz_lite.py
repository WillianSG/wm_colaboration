import numpy as np


def draw_graph3( networkx_graph, notebook=True, output_filename='graph.html', open_output=False,
                 show_buttons=False, only_physics_buttons=False ):
    # import
    from pyvis import network as net
    
    # make a pyvis network
    pyvis_graph = net.Network( notebook=notebook )
    pyvis_graph.width = '1000px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in networkx_graph.nodes( data=True ):
        pyvis_graph.add_node( node, **node_attrs )
    #         print(node,node_attrs)
    
    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in networkx_graph.edges( data=True ):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs[ 'value' ] = edge_attrs[ 'weight' ]
        # add the edge
        pyvis_graph.add_edge( source, target, **edge_attrs )
    
    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons( filter_=[ 'physics' ] )
        else:
            pyvis_graph.show_buttons()
    
    # return and also save
    pyvis_graph.show( output_filename )
    
    if open_output:
        import webbrowser
        import os
        
        webbrowser.open( f'file://{os.getcwd()}/{output_filename}' )


# Build NetworkX graph
import networkx as nx

pre_neurons = [ 7, 9, 16, 51, 57, 114, 122, 126, 129, 138, 170, 202, 208, 221, 254 ]
post_neurons = [ 7, 9, 16, 51, 57, 114, 122, 126, 129, 138, 170, 202, 208, 221, 254 ]
edges_pre = [ 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 51, 51, 51, 51, 51, 51, 57, 57, 57, 57, 57,
              57, 114, 114, 114, 114, 114, 114, 114, 114, 122, 122, 122, 126, 126, 126, 126, 126, 126, 129, 129, 129,
              129, 129, 138, 138, 138, 138, 138, 138, 170, 170, 170, 170, 170, 170, 202, 202, 202, 202, 202, 202, 202,
              208, 208, 208, 208, 208, 208, 221, 221, 254, 254, 254, 254, 254, 254, 254, 254 ]
edges_post = [ 122, 129, 170, 51, 122, 129, 138, 208, 7, 9, 51, 57, 114, 129, 170, 208, 221, 9, 129, 170, 202, 221, 254,
               114, 129, 170, 202, 208, 254, 51, 122, 126, 129, 138, 202, 208, 254, 114, 202, 208, 9, 51, 122, 202, 221,
               254, 51, 114, 126, 138, 170, 7, 57, 114, 126, 129, 170, 7, 9, 57, 126, 138, 221, 51, 114, 126, 129, 170,
               208, 221, 7, 51, 138, 170, 202, 254, 51, 170, 9, 16, 57, 114, 122, 126, 138, 170 ]
edges_weights = [ 0.0, 0.0, 0.0, 0.0075, 0.0, 0.0, 0.0, 0.0, 0.0075, 0.0075, 0.0075, 0.0075, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005,
                  0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0005,
                  0.0005, 0.0005, 0.0005, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0, 0.0, 0.0005, 0.0005,
                  0.0005, 0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005,
                  0.0, 0.0005, 0.0, 0.0, 0.0, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005 ]

edges_pre = [ k for i, k in enumerate( edges_pre ) if edges_weights[ i ] != 0 ]
edges_post = [ k for i, k in enumerate( edges_post ) if edges_weights[ i ] != 0 ]
edges_weights = [ k for i, k in enumerate( edges_weights ) if edges_weights[ i ] != 0 ]

g = nx.Graph()
g.add_nodes_from( pre_neurons )
g.add_nodes_from( post_neurons )
for i, j, w in zip( edges_pre, edges_post, edges_weights ):
    g.add_edge( i, j, weight=w )

a = draw_graph3( g, output_filename='test.html', open_output=True, notebook=False, show_buttons=True,
                 only_physics_buttons=True )
