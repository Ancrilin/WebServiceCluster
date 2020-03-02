import time
from datetime import timedelta
import pandas as pd
import os
import csv
import networkx as nx
from util.graph import Graph
import numpy as np


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_dataset(filepath):
    with open(filepath, 'r', encoding="utf-8") as fp:
        reader = csv.reader(fp)
        source = []
        for row in reader:
            source.append(row)
        source = source[1:]

    def prepare(source):
        print("preparing data...")
        positive_graph = nx.Graph()
        negative_graph = nx.Graph()
        for row in source:
            u = row[1].lower()
            v = row[2].lower()
            w = row[3]
            w = float(w)
            # if w < 0:
            #     print(u+"---"+v+"---"+str(w))
            if w > 0:
                positive_graph.add_edge(u, v, weight=w)
            if w < 0:
                negative_graph.add_edge(u, v, weight=w)
        print("ending preparation")
        print("positive_number_of_nodes: " + str(positive_graph.number_of_nodes()))
        print("positive_number_of_edges: " + str(positive_graph.number_of_edges()))
        print("negative_number_of_nodes: " + str(negative_graph.number_of_nodes()))
        print("negative_number_of_edges: " + str(negative_graph.number_of_edges()))
        return positive_graph, negative_graph

    positive_graph, negative_graph = prepare(source)
    my_graph = Graph(positive_graph, negative_graph)
    print("getting triplets...")
    del source
    triplets = my_graph.get_triplets()
    vocab = my_graph.vocab.getnode2id()
    return triplets, vocab


if __name__ == '__main__':
    # print(os.getcwd())
    # filepath1 = '../../data/20ClassesRawData_API_cleanTag.csv'
    # filepath2 = "../../data/WebNet_df.csv"
    # rawdata = pd.read_csv(filepath1, header=0)
    # data_raw = rawdata[['name', 'description', 'tags', 'category']].values
    # data = np.array([1,2,3,4])
    # d = np.array([3,4,5])
    triplets, vocab = get_dataset('../data/WebNet_df.csv')
    print(vocab)
