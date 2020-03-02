import networkx as nx
import numpy as np

class Vocabulary(object):
    def __init__(self, graph):
        self._id2node = {}
        self._node2id = {}
        self._curr_id = 1
        for node in graph.nodes():#遍历所有节点
            if node not in self._node2id:#如果该节点没有编号，加入
                self._curr_id += 1
                self._node2id[node] = self._curr_id#加入到节点-id字典中
                self._id2node[self._curr_id] = node#加入到id-节点字典中

    def id2node(self, id):
        return self._id2node[id]

    def node2id(self, node):
        return self._node2id[node]

    def augment(self, graph):#增加图
        for node in graph.nodes():
            if node not in self._node2id:
                self._curr_id += 1
                self._node2id[node] = self._curr_id
                self._id2node[self._curr_id] = node

    def __len__(self):
        return self._curr_id

    def getnode2id(self):
        return self._node2id


class Graph(object):
    def __init__(self, positive_graph, negative_graph):
        self.positive_graph = positive_graph
        self.negative_graph = negative_graph
        self.vocab = Vocabulary(positive_graph)
        self.vocab.augment(negative_graph)

    def get_positive_edges(self):
        return self.positive_graph.edges()

    def get_negative_edges(self):
        return self.negative_graph.edges()

    def __len__(self):
        return len(self.vocab)
        #return max(len(self.positive_graph), len(self.negative_graph))

    def get_triplets(self, p0=True, ids=True):
        triplets = []
        for xi in self.positive_graph.nodes():#遍历所有正节点
            for xj in self.positive_graph[xi]:#遍历该正节点的所有连接节点
                if xj in self.negative_graph:#如果是负节点
                    for xk in self.negative_graph[xj]:
                        a, b, c = xi, xj, xk
                        if ids:
                            a = self.vocab.node2id(xi)
                            b = self.vocab.node2id(xj)
                            c = self.vocab.node2id(xk)
                        triplets.append([a, b, c])
                elif p0:#没有负节点则增加虚拟节点
                    a, b = xi, xj
                    c = 0
                    if ids:
                        a = self.vocab.node2id(xi)
                        b = self.vocab.node2id(xj)
                    triplets.append([a, b, c])
        triplets = np.array(triplets)
        return triplets

    @staticmethod
    def read_from_file(filepath, delimiter=',', directed=False):
        positive_graph = nx.DiGraph() if directed else nx.Graph()#DiGraph是有向图的基类
        negative_graph = nx.DiGraph() if directed else nx.Graph()
        file = open(filepath)
        for line in file:
            line = line.strip()
            #print(line)
            u, v, w = line.split(delimiter)
            w = float(w)
            if w > 0:
                positive_graph.add_edge(u, v, weight=w)
            if w < 0:
                negative_graph.add_edge(u, v, weight=w)
        file.close()
        graph = Graph(positive_graph, negative_graph)
        return graph







