import networkx as nx
from recsys_framework.graph.walker import *
from recsys_framework.data_manager.reader.Movielens100KReader import Movielens100KReader
import recsys_framework.graph.graph_builder as gbuilder




if __name__ == '__main__':
    ### Create the bipartite graph associated to M100K
    reader = Movielens100KReader()
    m100k_dataset = reader.load_data()
    m100k_urm = m100k_dataset.get_URM()
    bip_graph = gbuilder.from_user_rating_matrix(m100k_urm)

    ### Select a node random walk starting from it
    walker = BiasedRandomWalker(bip_graph)
    rw_result = walker.run(nodes=[0], n=5, length=2)
    print(rw_result)