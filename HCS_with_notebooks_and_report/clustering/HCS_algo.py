import joblib
import numpy as np
import networkx as nx
import copy

from settings.general_settings import *
from settings.path_settings import *

# description: check if a graph is connected
# parameters:
    # G: nx.Graph, input graph whose connectivity should be checked
    # E: list, edge list of the graph G
def highly_connected(G: nx.Graph, E: list):
    return len(E) > (len(G.nodes)/2)


# description: remove a set of edges from a graph
# parameters:
    # G: nx.Graph, input graphs where the edges should be removed
    # E: list, set of edges to remove
def remove_edges(G: nx.Graph, E: list):
    for edge in E:
        G.remove_edge(*edge)
    return G


# description: function to perform the HCS clustering for one connected component
# parameters:
    # G: nx.Graph, connected input graph on which clustering should be performed
def HCS_algo_loop(G: nx.Graph):
    # perform the minium cut of the input graph using the stoer-wagner algorithm
    if len(list(G.nodes)) < 2:
        return G
    cut_value, V = nx.algorithms.connectivity.stoerwagner.stoer_wagner(G)
    E = list(nx.edge_boundary(G, V[0], V[1]))

    # if G is not highly connected, the minimum cut is removed
    if not highly_connected(G, E):
        G = remove_edges(G, E)
        sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if len(sub_graphs) == 2:
            # perform the algorithm iteratively on the two components obtained by removing the minium cut
            if len(list(sub_graphs[0].nodes)) > 1:
                H = HCS_algo_loop(G=sub_graphs[0])
            else:
                H = sub_graphs[0]
            if len(list(sub_graphs[1].nodes)) > 1:
                _H = HCS_algo_loop(G=sub_graphs[1])
            else:
                _H = sub_graphs[1]

            # put the results of the iterative procedures back together: the new G is then a disconnected graph
            G = nx.compose(H, _H)
    return G


# description: performs the HCS clustering on the input graph
# parameters:
    # G: nx.Graph, graph on which the clustering should be performed
def HCS_algo(G):
    # seperate the connected components of the input graph
    list_input = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    # perform the clustering on each of the connected components seperately
    list_results = [HCS_algo_loop(G=graph) for graph in list_input]
    return list_input, list_results


# description: function to get the clustering labels and perform the clustering
# parameters:
    # G: nx.Graph, graph for which the clustering labels need to be determined
def HCS_labels(G: nx.Graph):
    # perform the clustering using the functions from above
    list_input, list_results = HCS_algo(G)

    # initialize the values for the return of the class labels
    class_counter = 1
    list_labels = []
    labels = np.zeros(shape=len(G), dtype=np.uint16)

    # loop over the connected components in the input graph to name the clusters in ascending numbers
    for i in np.arange(len(list_input)):
        sub_graphs = (list_input[i].subgraph(c).copy() for c in nx.connected_components(list_results[i]))
        for _class, _cluster in enumerate(sub_graphs, class_counter):
            c = list(_cluster.nodes)
            labels[c] = _class

        # adjust the counting parameter and add the classes from this loop's connected component to the output list
        class_counter = _class + 1
        list_labels.append(labels)
    joblib.dump(labels, os.path.join(path_data + ''.join(['labels_sigma=', str(sigma), '_eps=', str(epsilon), '_weighted=', str(weighted), '.joblib'])))
    return labels
