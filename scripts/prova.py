#%%
import networkx as nx
from recsys_framework.graph.walker import *
from recsys_framework.data_manager.reader.Movielens100KReader import Movielens100KReader
import recsys_framework.graph.graph_builder as gbuilder
from gensim.models import Word2Vec

#%% md

### Create the bipartite graph associated to M100K

#%%

reader = Movielens100KReader()
m100k_dataset = reader.load_data()
m100k_urm = m100k_dataset.get_URM()
m100k_graph = gbuilder.from_user_rating_matrix(m100k_urm)

#%% md
### Perform biased RWs
#%%
walker = UniformRandomWalker(m100k_graph)

walks = walker.run(
    nodes=list(m100k_graph.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    #p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    #q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))
#%%
#model = Word2Vec(walks, size=128, window=5, min_count=0, sg=1, workers=2, iter=1)


