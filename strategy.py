'''
===========
   SIMULATOR USAGE
===========

>>> import sim
>>> sim.run([graph], [dict with keys as names and values as a list of nodes])

Returns a dictionary containing the names and the number of nodes they got.

Example:
>>> graph = {"2": ["6", "3", "7", "2"], "3": ["2", "7, "12"], ... }
>>> nodes = {"strategy1": ["1", "5"], "strategy2": ["5", "23"], ... }
>>> sim.run(graph, nodes)
>>> {"strategy1": 243, "strategy6": 121, "strategy2": 13}

Possible Errors:
- KeyError: Will occur if any seed nodes are invalid (i.e. do not exist on the
            graph).
'''

import sim
import json
import heapq
import numpy as np
import networkx as nx

def jsonToDict(filename):
    """
    Import a .json file as a dictionary
    """
    file = open(filename)
    d = json.load(file)
    return d

def degreeCentrality(graph, N):
    """
    Return the N nodes with the largest degree centrality in the graph

    Return a list of strings of nodes
    """
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree

    for node in graph.keys():
        if len(graph[node]) > topN[0][0]:
            heapq.heappushpop(topN, (len(graph[node]), node))

    ret = []
    for i in topN:
        ret.append(i[1])
    return ret

def eigenvectorCentrality(graph, N):
    """
    Return the N nodes with the largest eigenvector centrality in the graph

    Return a list of strings of nodes
    """
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree
    adjacency = [[0 for _ in range(len(graph))] for _ in range(len(graph))]

    for node in graph.keys():
        for neighbor in graph[node]:
            adjacency[int(node)][int(neighbor)] = 1
            adjacency[int(node)][int(neighbor)] = 1

    eigenvalues, eigenvectors = np.linalg.eig(adjacency)
    # only take the eigenvector of the largest eigenvalue
    largestEigen = list(eigenvalues).index(max(eigenvalues))
    score = eigenvectors[largestEigen]

    for node in range(len(score)):
        if score[node] > topN[0][0]:
            heapq.heappushpop(topN, (score[node], str(node)))

    ret = []
    for i in topN:
        ret.append(i[1])
    return ret

def betweennessCentrality(graph, N):
    """
    Return the N nodes with the largest betweenness centrality in the graph

    Uses the networkx package

    Return a list of strings of nodes
    """
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree

    G = nx.Graph()
    for node in graph.keys():
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    score = nx.betweenness_centrality(G)

    for node in score.keys():
        if score[node] > topN[0][0]:
            heapq.heappushpop(topN, (score[node], str(node)))

    ret = []
    for i in topN:
        ret.append(i[1])
    return ret

def closenessCentrality(graph, N):
    """
    Return the N nodes with the largest closeness centrality in the graph

    Uses the networkx package

    Return a list of strings of nodes
    """
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree

    G = nx.Graph()
    for node in graph.keys():
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    score = nx.closeness_centrality(G)

    for node in score.keys():
        if score[node] > topN[0][0]:
            heapq.heappushpop(topN, (score[node], str(node)))

    ret = []
    for i in topN:
        ret.append(i[1])
    return ret

def voteRank(graph, N):
    """
    Return the N nodes with the largest voterank in the graph

    Uses the networkx package

    Return a list of strings of nodes
    """
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree

    G = nx.Graph()
    for node in graph.keys():
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    return nx.voterank(G, N, max_iter=2000)

def clustering(graph, N):
    topN = [(-float("inf"), "null")] * N # initialize N seed nodes with -inf degree

    G = nx.Graph()
    for node in graph.keys():
        for neighbor in graph[node]:
            G.add_edge(node, neighbor)

    cluster = nx.algorithms.cluster.clustering(G)
    cluster = sorted(cluster)

    return cluster[:10]

def make_simulation(filename, N):
    """
    Run a simulation with 2 or more strategies for the given input graph
    """
    graph = jsonToDict(filename)
    print("{} total nodes.".format(len(graph)))

    ########### run a strategy here

    # use half of degree centrality
    # topN_1 = degreeCentrality(graph, N)
    # topN_2 = closenessCentrality(graph, N)
    #
    # topN = topN_1[5:]
    # i = 9
    # while len(topN) < 10:
    #     topN.append(topN_2[i])
    #     i -= 1
    # topN_2 = topN


    topN_1 = degreeCentrality(graph, N)
    if len(graph) > 2500:
        topN_2 = voteRank(graph, N) # less accurate, but shorter runtime
    else:
        topN_2 = closenessCentrality(graph, N) # more accurate

    ###########

    # simulate
    listOfStrategies = {"Degree":topN_1, "Hybrid":topN_2}
    results = sim.run(graph, listOfStrategies)
    print(results)

def make_submission(filename, N):
    """
    Run a strategy and make an output file

    Note: output file has to have N * 50 values to simulate 50 rounds
    """
    graph = jsonToDict(filename)
    print("{} total nodes.".format(len(graph)))

    ########### run a strategy here

    if len(graph) > 2500: # if big graph, use VoteRank (less good but shorter runtime)
        topN = voteRank(graph, N)
    else:
        topN = closenessCentrality(graph, N) # more accurate but runs longer than 5min for graphs with +2500 nodes

    ###########

    out = open("output.txt","w+")

    for round in range(50):
        for i in topN:
             out.write(i + "\n")
    out.close()

if __name__== "__main__":
    filename = "2.10.11.json" # CHANGE THIS

    N = int(filename.split(".")[1])
    print("Running {} with N = {}".format(filename, N))
    make_simulation(filename, N)
    #make_submission(filename, N)
