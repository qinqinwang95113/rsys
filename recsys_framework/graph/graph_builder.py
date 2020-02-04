import networkx as nx
from recsys_framework.graph.element_data import Node
from tqdm import tqdm
from networkx.convert_matrix import _generate_weighted_edges
import logging


def from_user_rating_matrix(urm, cell_value='weight', user_feature=None, item_feature=None):
    #TODO: TO TEST!
    """
    create an undirected bipartite graph starting from a user-rating matrix

    params:
    ------

    urm: User rating matrix from which create the graph

    cell_value: if weight the entries of the matrix will be considered the edges weight

    user_feature: pandas dataframe with user features, each column represent a different feature and it has to be
    accessible through the user idx

    item_feature: pandas dataframe with item features, each column represent a different feature and it has to be
    accessible through the item idx
    """

    #initialize an empty undirected graph
    G = nx.Graph()
    n_user, n_item = urm.shape

    # define the nodes list used to create the graph
    user_node_list = []
    item_node_list = []

    print('Adding users nodes...')
    for i in tqdm(range(n_user)):
        user_node_list.append((i, {'lookupid': i, 'kind': 'user'}))

    print('Adding item nodes...')
    for i in tqdm(range(n_user, n_user+n_item)):
        item_node_list.append((i, {'lookupid': i-n_user, 'kind': 'item'}))

    G.add_nodes_from(user_node_list)
    G.add_nodes_from(item_node_list)

    # Create an iterable over (u, v, w) triples and for each triple, add an
    # edge from u to v with weight w.
    triples = ((u, n_user+v, d) for (u, v, d) in _generate_weighted_edges(urm))

    G.add_weighted_edges_from(triples, weight='weight')
    return G