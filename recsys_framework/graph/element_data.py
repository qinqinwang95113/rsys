from abc import abstractmethod, ABC

class Node(ABC):
    """
    Abstract class for graph nodes
    """

    def __init__(self, idx, **args):
        self.idx = idx

        #set the additional attribute passed
        for k, v in args.items():
            self.k = v



class Edge(ABC):
    """
    Abstract class for graph edges
    """

if __name__ == '__main__':
    prova = Node(3, cane=4, pippo=5)
    a=4