import networkx as nx
import numpy as np
from collections import defaultdict


# TODO what's the degree distribution of the graph?  is it well-formed? (Gaussian, exponential, or power-law)
# TODO The RCN model start from a random graph, but is it the most realistic setting? Maybe a geometric graph would
#  be more realistic?

def timefunc(func):
    import time
    import functools

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"Function: {func.__name__}, Time: {time_elapsed}")
        return result

    return time_closure


def rgb_to_hex(r, g, b):
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)


def hex_to_rgb(h):
    h = h.lstrip('#')

    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rcn2nx(rcn,
           neurons_subsample=None, edges_subsample=None, subsample_attractors=False, seed=None,
           remove_edges_threshold=0.0, output_filename='graph'):
    """Given an instance of a RecurrentCompetitiveNet builds the corresponding NetworkX graph.
    
    Parameters:
    net (RecurrentCompetitiveNet): the Brian2 net of which to build the graph
    neurons_subsample (float): percentage of neurons to sample
    edges_subsample (float): percentage of edges to sample
    subsample_attractors (bool): ad-hoc sampling from attractor A, attractor B, and non-specific population
    seed (int): set seed to make the subsampling reproducible
    remove_edges_threshold (float): remove edges whose weight is smaller than remove_edges_threshold from graph
    output_filename (str): the name of the file to save to disk in os.getcwd()
    
    Returns:
    networkx.DiGraph: The graph build from net
    
    """
    import os

    # ------ Subsample graph nodes
    rng = np.random.default_rng(seed)

    def closestNumber(n, m):
        """Function to find the number closest to n and divisible by m"""

        # Find the quotient
        q = int(n / m)

        # 1st possible closest number
        n1 = m * q

        # 2nd possible closest number
        if ((n * m) > 0):
            n2 = (m * (q + 1))
        else:
            n2 = (m * (q - 1))

        # if true, then n1 is the required closest number
        if (abs(n - n1) < abs(n - n2)):
            return n1

        # else n2 is the required closest number
        return n2

    if neurons_subsample:
        assert neurons_subsample > 0 and neurons_subsample <= 1

        e_neurons_subsample = round(len(rcn.E) * neurons_subsample)
        i_neurons_subsample = round(len(rcn.I) * neurons_subsample)

        if subsample_attractors:
            e_neurons_subsample = closestNumber(e_neurons_subsample, 3)

            e_neurons = np.concatenate(
                [rng.choice(range(0, 64), int(e_neurons_subsample / 3), replace=False),
                 rng.choice(range(100, 164), int(e_neurons_subsample / 3), replace=False),
                 rng.choice(range(164, 256), int(e_neurons_subsample / 3), replace=False)]
            )
            i_neurons = rng.choice(range(0, len(rcn.I)), i_neurons_subsample, replace=False)
        else:
            e_neurons = rng.choice(range(0, len(rcn.E)), e_neurons_subsample, replace=False)
            i_neurons = rng.choice(range(0, len(rcn.I)), i_neurons_subsample, replace=False)
    else:
        e_neurons = np.array(range(0, len(rcn.E)))
        i_neurons = np.array(range(0, len(rcn.I)))

    # ----- Make sure that there are no duplicates in neurons subsample or Brian2 indexing runs into a bug when slicing
    assert len(np.unique(e_neurons)) == len(e_neurons) or len(np.unique(i_neurons)) == len(i_neurons)
    # ----- Add Excitatory-to-Excitatory connections
    e2e_edges_pre = rcn.E_E.i[e_neurons, e_neurons].tolist()
    e2e_edges_post = rcn.E_E.j[e_neurons, e_neurons].tolist()
    e2e_edges_weight = rcn.E_E.w_[e_neurons, e_neurons].tolist()
    e2e_edges = np.vstack((e2e_edges_pre, e2e_edges_post, e2e_edges_weight))
    # ----- Add Inhibitory-to-Excitatory connections
    i2e_edges_pre = rcn.I_E.i[i_neurons, e_neurons].tolist()
    i2e_edges_post = rcn.I_E.j[i_neurons, e_neurons].tolist()
    i2e_edges_weight = rcn.I_E.w_[i_neurons, e_neurons].tolist()
    i2e_edges = np.vstack((i2e_edges_pre, i2e_edges_post, i2e_edges_weight))
    # ----- Add Excitatory-to-Inhibitory connections
    e2i_edges_pre = rcn.E_I.i[e_neurons, i_neurons].tolist()
    e2i_edges_post = rcn.E_I.j[e_neurons, i_neurons].tolist()
    e2i_edges_weight = rcn.E_I.w_[e_neurons, i_neurons].tolist()
    e2i_edges = np.vstack((e2i_edges_pre, e2i_edges_post, e2i_edges_weight))
    # ----- Add Inhibitory-to-Inhibitory connections
    i2i_edges_pre = rcn.I_I.i[i_neurons, i_neurons].tolist()
    i2i_edges_post = rcn.I_I.j[i_neurons, i_neurons].tolist()
    i2i_edges_weight = rcn.I_I.w_[i_neurons, i_neurons].tolist()
    i2i_edges = np.vstack((i2i_edges_pre, i2i_edges_post, i2i_edges_weight))

    if edges_subsample:
        assert edges_subsample > 0 and edges_subsample <= 1

        e2e_edges_subsample = round(len(e2e_edges_pre) * edges_subsample)
        e2e_edges = rng.choice(e2e_edges, e2e_edges_subsample, replace=False, axis=1)
        i2e_edges_subsample = round(len(i2e_edges_pre) * edges_subsample)
        i2e_edges = rng.choice(i2e_edges, i2e_edges_subsample, replace=False, axis=1)
        e2i_edges_subsample = round(len(e2i_edges_pre) * edges_subsample)
        e2i_edges = rng.choice(e2i_edges, e2i_edges_subsample, replace=False, axis=1)
        i2i_edges_subsample = round(len(i2i_edges_pre) * edges_subsample)
        i2i_edges = rng.choice(i2i_edges, i2i_edges_subsample, replace=False, axis=1)

    if remove_edges_threshold is not None:
        e2e_edges = e2e_edges[:, e2e_edges[2] > remove_edges_threshold]
        i2e_edges = i2e_edges[:, i2e_edges[2] > remove_edges_threshold]
        e2i_edges = e2i_edges[:, e2i_edges[2] > remove_edges_threshold]
        i2i_edges = i2i_edges[:, i2i_edges[2] > remove_edges_threshold]

    g = nx.DiGraph()

    # Add excitatory neurons
    e_nodes = [f'e_{i}' for i in e_neurons]
    e_neuron_spikes = np.array(rcn.E_mon.count)[e_neurons]
    for n, spk in zip(e_nodes, e_neuron_spikes):
        g.add_node(n, label=n, color='rgba(0,0,255,1)', title=' ', type='excitatory', activity=int(spk))
    # Add synapses using excitatory neurons
    for col in e2e_edges.T:
        g.add_edge(f'e_{int(col[0])}', f'e_{int(col[1])}', weight=col[2])

    # Classify excitatory neurons
    tag_weakly_connected_components(g)
    tag_attracting_components(g)
    colour_by_attractor(g)

    # Add inhibitory neurons
    i_nodes = [f'i_{i}' for i in i_neurons]
    i_neuron_spikes = np.array(rcn.I_mon.count)[i_neurons]
    for n, spk in zip(i_nodes, i_neuron_spikes):
        g.add_node(n, label=n, color='rgba(255,0,0,0.5)', title=' ', type='inhibitory', activity=int(spk))
    # Add synapses using inhibitory neurons
    for col in i2e_edges.T:
        g.add_edge(f'i_{int(col[0])}', f'e_{int(col[1])}', weight=col[2])
    for col in e2i_edges.T:
        g.add_edge(f'e_{int(col[0])}', f'i_{int(col[1])}', weight=col[2])
    for col in i2i_edges.T:
        g.add_edge(f'i_{int(col[0])}', f'i_{int(col[1])}', weight=col[2])

    # GraphML does not support list or any non-primitive type
    if output_filename:
        nx.write_graphml(g, f'{os.getcwd()}/{output_filename}.graphml')

    return g


def nx2pyvis(input, notebook=False, output_filename='graph',
             scale_by='e-i balance',
             neuron_types=None, synapse_types=None,
             open_output=True, show_buttons=True, only_physics_buttons=True, window_size=(1000, 1000)):
    """Given an instance of a Networx.Graph builds the corresponding PyVis graph for display purposes.

    Parameters:
    input (nx.Graph, RecurrentCompetitiveNet, or str): the NetworkX graph or the RecurrentCompetitiveNet to convert
    to PyVis, or the path to a GraphML file to convert
    output_filename (str): the name of the file HTML to save to disk in os.getcwd()
    scale_by (str): the node parameter whose value is used to visually scale the nodes.  Can be 'neighbours',
    'excitation', 'inhibition','e-i balance', 'activity'
    neural_population (set of str): which neural populations to include.  For example, 'e_e' includes
        excitatory-to-excitatory connections, 'i_e' includes inhibitory-to-excitatory connections
    synapse_types (set of str): which synapse types to include.  'e' includes excitatory neurons, 'i' inhibitory ones
    open_output (bool): set to True to open the HTML file in the default browser
    show_buttons (bool): set to true to display the PyVis configuration UI in the HTML
    only_physics_buttons (bool): set to true to display the physics PyVis configuration UI in the HTML
    windows_size (tuple): the width and height of the window to display the graph in the HTML

    Returns:
    None

    """
    from pyvis import network as net
    import pathlib
    from collections import defaultdict
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet

    if neuron_types is None:
        neuron_types = {'e', 'i'}
    if synapse_types is None:
        synapse_types = {'e_e', 'i_e', 'e_i', 'i_i'}

    if isinstance(input, nx.Graph):
        nx_graph = input
    elif isinstance(input, RecurrentCompetitiveNet):
        nx_graph = rcn2nx(input)
    elif isinstance(input, str) and pathlib.Path(input).suffix == '.graphml':
        nx_graph = nx.read_graphml(input)

    # make a pyvis network
    pyvis_graph = net.Network(notebook=notebook, directed=True)
    pyvis_graph.width = f'{window_size[0]}px'
    pyvis_graph.height = f'{window_size[1]}px'
    # for each node and its attributes in the networkx graph
    for node, node_attrs in nx_graph.nodes(data=True):
        # remove nodes that aren't of included type
        include = False
        for n in neuron_types:
            if node.split('_')[0] == n:
                include = True
                break
        if not include:
            continue

        pyvis_graph.add_node(node, **node_attrs)

    # for each edge and its attributes in the networkx graph
    for source, target, edge_attrs in nx_graph.edges(data=True):
        # remove edges that aren't of included type
        if (source.split('_')[0] in neuron_types and target.split('_')[0] in neuron_types) and (
                source.split('_')[0] + '_' + target.split('_')[0] in synapse_types):
            # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                # place at key 'value' the weight of the edge
                if 'e_' in source and 'e_' in target:
                    edge_attrs['value'] = edge_attrs['weight']
                else:
                    edge_attrs['value'] = ''
            # add the edge
            pyvis_graph.add_edge(source, target, **edge_attrs, title='', arrows='to', dashes=True)

    # add neighbour data and statistics to node hover popup
    type_colours = {'excitatory': 'blue', 'inhibitory': 'red'}
    for node in pyvis_graph.nodes:
        node_pred = [n for n in nx_graph.predecessors(node['id'])]
        node_pred_exc = [n for n in node_pred if 'e_' in n]
        node_pred_inh = [n for n in node_pred if 'i_' in n]
        node_succ = [n for n in nx_graph.successors(node['id'])]
        node_succ_exc = [n for n in node_succ if 'e_' in n]
        node_succ_inh = [n for n in node_succ if 'i_' in n]

        # associate each incoming attractor with its effect on the current node
        attractor_pred_exc = defaultdict(float)
        for w, act, atr in zip(get_weights(nx_graph, node_pred_exc, node['id']),
                               [nx_graph.nodes(data=True)[n]['activity'] for n in node_pred_exc],
                               [nx_graph.nodes(data=True)[n]['attractor'] for n in node_pred_exc]):
            attractor_pred_exc[atr] += float(w * act)
        total_excitation = sum(attractor_pred_exc.values())
        # compute inhibition acting on this node
        total_inhibition = np.sum(get_weights(nx_graph, node_pred_inh, node['id']) * np.array(
            [nx_graph.nodes(data=True)[n]['activity'] for n in node_pred_inh]))

        # compute the effect this node has on attractors
        attractor_succ_exc = sorted(
            np.array(np.unique([nx_graph.nodes(data=True)[n]['attractor'] for n in node_succ_exc],
                               return_counts=True)).T,
            key=lambda x: x[1], reverse=True)
        # compute the effect attractors have on this node
        attractor_pred_exc = sorted(attractor_pred_exc.items(), key=lambda x: x[1], reverse=True)

        attractor_colours = {n[1]['attractor']: n[1]['color'] for n in nx_graph.nodes(data=True) if
                             'e_' in n[0]}

        node['title'] += f'<center><h3>{node["id"]}</h3></center>'
        node['title'] += f'<h4 style="color: {"blue" if "e_" in node["id"] else "red"}">Kind: {node["type"]}</h4>'
        if 'e_' in node['id']:
            node['title'] += f'<h4 style="color: {node["color"]}">Attractor: {node["attractor"]}</h4>'
        node['title'] += f'<h4>E-I balance: {total_excitation - total_inhibition}</h4>'
        node['title'] += f'<h4># spikes: {node["activity"]}</h4>'
        node['title'] += f'<h3>Excitation (total {total_excitation}):</h3>'
        verb = 'Excites' if 'e_' in node['id'] else 'Inhibits'
        node['title'] += f'<h4>{verb} attractors (total neurons: {len(node_succ_exc)}):</h4>' + ' '.join(
            [f'<p style="color: {attractor_colours[a[0]]}">{a[0]} ({a[1]})</p>' for a in
             attractor_succ_exc]
        )
        node['title'] += f'<h4>Excited by attractors (total neurons: {len(node_pred_exc)}):</h4>' + ' '.join(
            [f'<p style="color: {attractor_colours[a[0]]}">{a[0]} ({a[1]})</p>' for a in
             attractor_pred_exc]
        )
        node['title'] += f'<h3>Inhibition (total {total_inhibition}):</h3>'
        node['title'] += f'<h4>{verb} inhibitory ({len(node_succ_inh)}):</h4>' + ' '.join(
            [f'<p style="color: red">{n}</p>' for n in node_succ_inh]
        )
        node['title'] += f'<h4>Inhibited by ({len(node_pred_inh)}):</h4>' + ' '.join(
            [f'<p style="color: red">{n}</p>' for n in node_pred_inh]
        )

        if scale_by == 'neighbours':
            node['value'] = len(node_pred + node_succ)
        elif scale_by == 'excitation':  # # more excitation, larger node
            node['value'] = total_excitation
        elif scale_by == 'inhibition':  # more inhibition, smaller node
            node['value'] = 1 - total_inhibition
        elif scale_by == 'e-i balance':
            node['value'] = total_excitation - total_inhibition
        elif scale_by == 'activity':
            node['value'] = node['activity']

    for edge in pyvis_graph.edges:
        edge['title'] += f'<center><h3 style="color: ' \
                         f'{type_colours[nx_graph.nodes[edge["from"]]["type"]]}">{edge["from"]} → ' \
                         f'{edge["to"]}</h3></center>'
        edge['title'] += f'<p style="color: {type_colours[nx_graph.nodes[edge["from"]]["type"]]}">' \
                         f'Kind: {nx_graph.nodes[edge["from"]]["type"]}</p>'
        edge['title'] += 'Weight: ' + str(edge['weight'])

    # turn buttons on
    if show_buttons:
        if only_physics_buttons:
            pyvis_graph.show_buttons(filter_=['physics'])
        else:
            pyvis_graph.show_buttons()

    pyvis_graph.set_edge_smooth('dynamic')
    pyvis_graph.toggle_hide_edges_on_drag(True)
    pyvis_graph.force_atlas_2based(damping=0.7)

    # return and also save
    pyvis_graph.show(f'{output_filename}.html')

    if open_output:
        import webbrowser
        import os

        webbrowser.open(f'file://{os.getcwd()}/{output_filename}.html')


def colour_by_attractor(g):
    """Compute and store a different colour for each attractor in the NetworkX graph.
    
    Parameters:
    g (networkx.Graph): the graph for which to compute the attractor colour
    
    Returns:
    None
    
    """
    import cmasher as cmr

    num_attractors = len(set(nx.get_node_attributes(g, 'attractor').values()))
    colours = cmr.take_cmap_colors('tab20', num_attractors, return_fmt='int')

    for n, v in g.nodes(data=True):
        col = list(colours[g.nodes[n]['attractor']])
        g.nodes[n]['color'] = f'rgba({col[0]},{col[1]},{col[2]},1)'


def tag_weakly_connected_components(g):
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
    idx_components = {n: i for i, node_set in enumerate(nx.weakly_connected_components(g)) for n in node_set if
                      'e_' in n}
    for n, i in idx_components.items():
        g.nodes[n]['attractor'] = i


def tag_attracting_components(g):
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

    attracting_map = {}
    for atr in set(nx.get_node_attributes(g, 'attractor').values()):
        try:
            atr_nodes = [n for n, v in g.nodes(data=True) if v['attractor'] == atr]
            atr_subgraph = g.subgraph(atr_nodes)
            is_attracting = is_attracting_component(atr_subgraph)
            for n in atr_nodes:
                attracting_map[n] = {'is_attracting': is_attracting}
        except:
            continue

    nx.set_node_attributes(g, attracting_map)


def get_weights(nx_graph, source, target):
    import numbers

    if isinstance(source, numbers.Number) or isinstance(source, str):
        target = [source]
    if isinstance(target, numbers.Number) or isinstance(target, str):
        target = [target]

    weights = np.zeros((len(source), len(target)))

    for i, s in enumerate(source):
        for j, t in enumerate(target):
            try:
                weights[i, j] = nx_graph.get_edge_data(s, t)['weight']
            except:
                pass

    return weights


def check_input(input):
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet

    if not isinstance(input, nx.Graph) and isinstance(input, RecurrentCompetitiveNet):
        return rcn2nx(input)
    elif isinstance(input, nx.Graph):
        return input
    else:
        raise ValueError('input must be of type nx.Graph or RecurrentCompetitiveNet')


@timefunc
def attractor_excitation_statistics(input, statistic,
                                    include_weights=False, include_activity=False, normalise=True,
                                    comment=''):
    """Compute the attracting components in the NetworkX graph and store the information in the nodes.

    An attracting component in a directed graph G is a strongly connected component with the property that a random
    walker on the graph will never leave the component, once it enters the component.  The nodes in attracting
    components can also be thought of as recurrent nodes. If a random walker enters the attractor containing the node,
    then the node will be visited infinitely often.

    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the statistic
    statistic (str): the statistic to compute.  One between "inhibition", "excitation", "self-excitation'.
    include_weights (bool): whether to account for the edge weights in the calculation
    include_activity (bool): whether to account for the neuronal activity as spikes in the calculation
    normalise (bool): whether to normalise the computed statistic
    comment (str): the comment to append to the generated txt file in os.getcwd()

    Returns:
    None

    """

    import os
    from pprint import pprint
    from collections import defaultdict

    if statistic == 'inhibition':
        pred_nodes = lambda i, atr: 'i_' in i
        output_filename = 'attractor_inhibition'
        file_header = """Generated via the graphing/attractor_statistics function.
			
	Compute the inhibition of each attractor in the NetworkX graph as the sum of inhibitory neuron activity times
    the connection weight.
	This should be useful to quantify the degree of inhibition of each attractor.
	
	Results:
	
	"""
    elif statistic == 'excitation':
        pred_nodes = lambda i, atr: 'e_' in i
        output_filename = 'attractor_excitation'
        file_header = """Generated via the graphing/attractor_statistics function.
			
    Compute the excitation of each attractor in the NetworkX graph as the sum of excitatory neuron activity times
    the connection weight.
	This should be useful to quantify the degree of excitation of each attractor.
	
	Results:
	
	"""
    elif statistic == 'self-excitation':
        pred_nodes = lambda i, atr: 'e_' in i and atr == g.nodes(data=True)[i]['attractor']
        output_filename = 'attractor_self_excitation'
        file_header = """Generated via the graphing/attractor_statistics function.
			
    Compute the self-excitation of each attractor in the NetworkX graph as the sum of activity of excitatory
    neurons within the attractor times the connection weight.
	This should be useful to quantify the degree of excitation of each attractor.
	
	Results:
	
	"""

    g = check_input(input)

    attractor_statistic_amount = defaultdict(float)
    for atr in set(nx.get_node_attributes(g, 'attractor').values()):
        attractor_nodes = [n for n, v in g.nodes(data=True) if 'e_' in n and v['attractor'] == atr]

        for n in attractor_nodes:
            pred_n = [i for i in list(g.predecessors(n)) if pred_nodes(i, atr)]
            if include_weights:
                w = get_weights(g, n, pred_n)
            else:
                w = np.ones(len(pred_n))
            if include_activity:
                a = [g.nodes(data=True)[n]['activity'] for n in pred_n]
            else:
                a = np.ones(len(pred_n))
            attractor_statistic_amount[atr] += np.sum(np.array(w) * np.array(a))

    if normalise:
        norm = max(attractor_statistic_amount.values())
        attractor_statistic_amount = {k: v / norm for k, v in attractor_statistic_amount.items()}

    if not os.path.exists(f'{os.getcwd()}/{output_filename}.txt'):
        with open(f'{os.getcwd()}/{output_filename}.txt', 'w') as f:
            f.write(file_header)
            f.write(f'\t{comment}\n\t')
            pprint(attractor_statistic_amount, stream=f)
    else:
        with open(f'{os.getcwd()}/{output_filename}.txt', 'a') as f:
            f.write(f'\t{comment}\n\t')
            pprint(attractor_statistic_amount, stream=f)

    return attractor_statistic_amount


def attractor_connectivity_statistics(input, statistic,
                                      comment=''):
    """Computes the average node connectivity within each attractor in the NetworkX graph.
    This should be useful to quantify the amount of self-excitation each attractor has.
    The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G.
    Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
    along with their incident edges) to disconnect them.
    This functions is likely to be slow on the full network ( ~4 minutes on Apple M1).
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    connectivity
    approximate (bool): whether to use the approximate version of the average node connectivity
    comment (str): the comment to append to the generated txt file in os.getcwd()

    Returns:
    attractor_connectivity_amount (dict): dictionary of attractors with the average node connectivity as the value
    
"""
    import os
    from pprint import pprint
    from networkx.algorithms.connectivity.connectivity import average_node_connectivity, node_connectivity
    from networkx.algorithms.flow import shortest_augmenting_path
    from networkx.algorithms import approximation as approx

    output_filename = 'attractor_connectivity'

    g = check_input(input)

    attractor_connectivity_amount = {}

    for atr in set(nx.get_node_attributes(g, 'attractor').values()):
        attractor_nodes = [n for n, v in g.nodes(data=True) if 'e_' in n and v['attractor'] == atr]
        subgraph = g.subgraph(attractor_nodes)

        if statistic == 'connectivity':
            attractor_connectivity_amount[atr] = approx.node_connectivity(subgraph)
        elif statistic == 'approximate connectivity':
            attractor_connectivity_amount[atr] = node_connectivity(subgraph, flow_func=shortest_augmenting_path)
        elif statistic == 'cycles':
            attractor_connectivity_amount[atr] = len(list(nx.simple_cycles(subgraph)))

    if not os.path.exists(f'{os.getcwd()}/{output_filename}.txt'):
        with open(f'{os.getcwd()}/{output_filename}.txt', 'w') as f:
            f.write("""Generated via the graphing/attractor_connectivity function.
			
	Computes the average node connectivity within each attractor in the NetworkX graph.
	This should be useful to quantify the amount of self-excitation each attractor has.
	The average connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G.
	Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed (
	along with their incident edges) to disconnect them.
	
	Results:

	""")
            f.write(f'\t{comment}\n')
            pprint(attractor_connectivity_amount, stream=f)

    else:
        with open(f'{os.getcwd()}/{output_filename}.txt', 'a') as f:
            f.write(f'\t{comment}\n')
            pprint(attractor_connectivity_amount, stream=f)

    return attractor_connectivity_amount


def harcoded_attractor_algebraic_connectivity(input, variant=1):
    import os
    import csv

    from algebraic_connectivity_directed.algebraic_connectivity_directed import algebraic_connectivity_directed_variants
    from brian2 import second

    g = check_input(input)

    current_time = input.net.x / second

    # -- define attractor indices
    A1 = list(range(0, 64))
    A2 = list(range(100, 164))

    attractor_nodes_A1 = [n for n, v in g.nodes(data=True) if 'e_' in n and int(n.split('_')[1]) in A1]
    subgraph_A1 = g.subgraph(attractor_nodes_A1)
    algebraic_connectivity_A1 = algebraic_connectivity_directed_variants(subgraph_A1, variant)

    attractor_nodes_A2 = [n for n, v in g.nodes(data=True) if 'e_' in n and int(n.split('_')[1]) in A2]
    subgraph_A2 = g.subgraph(attractor_nodes_A2)
    algebraic_connectivity_A2 = algebraic_connectivity_directed_variants(subgraph_A2, variant)

    filename = '../graph_analysis/a_conn.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'A1', 'A2'])
            writer.writerow([current_time, algebraic_connectivity_A1, algebraic_connectivity_A2])
    else:
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, algebraic_connectivity_A1, algebraic_connectivity_A2])

    return current_time, algebraic_connectivity_A1, algebraic_connectivity_A2


def attractor_algebraic_connectivity(input, variant=1, comment=''):
    """Computes the directed algebraic connectivity for each attractor in graph G, based on the definitions in
    [C. W. Wu, "Synchronization in Complex Networks of Nonlinear Dynamical Systems", World Scientific, 2007].
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    algebraic connectivity
    variant (int): the variant of algebraic connectivity to use
    comment (str): the comment to append to the generated txt file in os.getcwd()

    Returns:
    attractor_statistic_amount (dict): dictionary of attractors with the average node connectivity as the value
    
"""
    import os
    from pprint import pprint

    from algebraic_connectivity_directed.algebraic_connectivity_directed import algebraic_connectivity_directed_variants
    output_filename = 'attractor_algebraic_connectivity'

    g = check_input(input)

    attractor_connectivity_amount = {}

    for atr in set(nx.get_node_attributes(g, 'attractor').values()):
        attractor_nodes = [n for n, v in g.nodes(data=True) if 'e_' in n and v['attractor'] == atr]
        subgraph = g.subgraph(attractor_nodes)

        attractor_connectivity_amount[atr] = algebraic_connectivity_directed_variants(subgraph, variant)

    if not os.path.exists(f'{os.getcwd()}/{output_filename}.txt'):
        with open(f'{os.getcwd()}/{output_filename}.txt', 'w') as f:
            f.write("""Computes the directed algebraic connectivity for each attractor in graph G, based on the
            definitions in
    [C. W. Wu, "Synchronization in Complex Networks of Nonlinear Dynamical Systems", World Scientific, 2007].
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    algebraic connectivity
    variant (int): the variant of algebraic connectivity to use
    comment (str): the comment to append to the generated txt file in os.getcwd()

    Returns:
    attractor_connectivity_amount (dict): dictionary of attractors with the algebraic connectivity as the value
    
""")
            f.write(f'\t{comment}\n')
            pprint(attractor_connectivity_amount, stream=f)

    else:
        with open(f'{os.getcwd()}/{output_filename}.txt', 'a') as f:
            f.write(f'\t{comment}\n')
            pprint(attractor_connectivity_amount, stream=f)

    return attractor_connectivity_amount


def attractor_mutual_inhibition(input,
                                include_weights=False, include_activity=False, normalise=True,
                                comment=''):
    """Computes how much each attractor inhibits the others in the NetworkX graph.
    
    Parameters:
    input (networkx.Graph or RecurrentCompetitiveNet): the graph or the RCN for which to compute the attractors'
    connectivity
    include_weights (bool): whether to account for the edge weights in the calculation
    include_activity (bool): whether to account for the neuronal activity as spikes in the calculation
    comment (str): the comment to append to the generated txt file in os.getcwd()

    Returns:
    attractor_inhibition_amount (dict): dictionary of attractors with the average node connectivity as the value
    
"""
    import os
    from pprint import pprint

    output_filename = 'attractor_mutual_inhibition'

    g = check_input(input)

    attractors = set(nx.get_node_attributes(g, 'attractor').values())
    attractor_inhibition_amount = defaultdict(dict)
    for source_atr in attractors:
        source_attractor_nodes = [n for n, v in g.nodes(data=True)
                                  if 'e_' in n and v['attractor'] == source_atr]

        # find successor inhibitory nodes from source attractor
        succ_inh = set()
        for n in source_attractor_nodes:
            succ_inh.update([i for i in list(g.successors(n)) if 'i_' in i])

        # check current attractor against all others
        for target_atr in attractors:
            target_attractor_nodes = [n for n, v in g.nodes(data=True) if
                                      'e_' in n and v['attractor'] == target_atr]

            # find predecessor inhibitory nodes from drain attractor
            pred_inh = set()
            for n in target_attractor_nodes:
                pred_inh.update([i for i in list(g.predecessors(n)) if 'i_' in i])

            inh = list(succ_inh.intersection(pred_inh))

            w = get_weights(g, inh, target_attractor_nodes)
            if not include_weights:
                w[w != 0] = 1
            a = np.array([g.nodes(data=True)[n]['activity'] for n in inh])
            if not include_activity:
                a[a != 0] = 1

            if include_weights or include_activity:
                attractor_inhibition_amount[source_atr][target_atr] = np.sum(w.T @ a) if len(inh) > 0 else 0
            else:
                attractor_inhibition_amount[source_atr][target_atr] = len(inh)

    if normalise:
        norm = max({k: max(v.values()) for k, v in attractor_inhibition_amount.items()}.values())
        normalised_amount = defaultdict(dict)
        for k, v in attractor_inhibition_amount.items():
            for kk, vv in v.items():
                normalised_amount[k][kk] = vv / norm
        attractor_inhibition_amount = normalised_amount

    if not os.path.exists(f'{os.getcwd()}/{output_filename}.txt'):
        with open(f'{os.getcwd()}/{output_filename}.txt', 'w') as f:
            f.write("""Generated via the graphing/attractor_mutual_inhibition function.
			
    Computes how much each attractor inhibits the others in the NetworkX graph.
	
	Results:

	""")
            f.write(f'\t{comment}\n')
            pprint(attractor_inhibition_amount, stream=f)

    else:
        with open(f'{os.getcwd()}/{output_filename}.txt', 'a') as f:
            f.write(f'\t{comment}\n')
            pprint(attractor_inhibition_amount, stream=f)

    return attractor_inhibition_amount


files = [
    'initial.html', 'initial.graphml', 'initial_complete.graphml',
    'first.html', 'first.graphml', 'first_complete.graphml',
    'second.html', 'second.graphml', 'second_complete.graphml',
    'rcn_population_spiking.png',
    'attractor_inhibition.txt', 'attractor_connectivity.txt', 'attractor_algebraic_connectivity.txt',
    'attractor_excitation.txt', 'attractor_self_excitation.txt',
    'E_spikes.txt', 'attractor_synchronisation.pdf', 'a_conn.csv', 'attractor_video.mp4'
]


def save_graph_results(folder='interesting_graph_results', additional_files=None, comments=''):
    """Saves files resulting from graphing in a folder

    Parameters:
    folder (str): the base folder under which results will be saved
    additional_files (list): any additional files to move along with the default ones
    comments (str): any comments to add to the folder.  Will be written into `comments.txt`

    Returns:
    None

"""
    import shutil
    import datetime
    import os

    if additional_files is None:
        additional_files = []

    new_folder = './' + folder + '/' + str(datetime.datetime.now().date()) + '_' + \
                 str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '.')
    os.makedirs(new_folder)

    files_local = files + additional_files

    count = 0
    for f in files_local:
        try:
            shutil.move(os.getcwd() + '/' + f, new_folder)
            count += 1
        except:
            continue

    if count > 0:
        print(f'Moved {count} files to {new_folder}')

        if comments:
            with open(f'{new_folder}/comments.txt', 'w') as f:
                f.write(comments)
        else:
            with open(f'{new_folder}/comments.txt', 'w') as f:
                f.write('Nothing particular to remark.')
    else:
        os.rmdir(new_folder)
        print('No files to move')


def clean_folder(additional_files=None):
    import os

    if additional_files is None:
        additional_files = []

    files_local = files + additional_files

    count = 0
    for f in files_local:
        try:
            os.remove(f)
            count += 1
        except:
            continue

    print(f'Removed {count} leftover files from {os.getcwd()}')
