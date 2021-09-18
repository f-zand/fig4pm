# METHODS FOR NETWORKX GRAPH CREATION
from general_methods import ranking_dict, adjacency_matrix_directed, adjacency_matrix_undirected, markov_chain_adjacency_matrix
import networkx as nx

# Create directed graph
def create_directed_graph(log):
    DG = nx.DiGraph()
    matrix = adjacency_matrix_directed(log)
    number_for_event = ranking_dict(log)
    event_for_number = dict((y, x) for x, y in number_for_event.items())
    for row in range(len(matrix)):
        for event in range(len(matrix[row])):
                if matrix[row][event] > 0:
                    DG.add_edge(event_for_number[row], event_for_number[event])
    return DG


# Create undirected graph
def create_undirected_graph(log):
    DG = nx.Graph()
    matrix = adjacency_matrix_undirected(log)
    number_for_event = ranking_dict(log)
    event_for_number = dict((y, x) for x, y in number_for_event.items())
    for row in range(len(matrix)):
        for event in range(len(matrix[row])):
            if matrix[row][event] > 0:
                DG.add_edge(event_for_number[row], event_for_number[event])
                DG.add_edge(event_for_number[event], event_for_number[row])
    return DG


# Create directed weighted graph
def create_directed_weighted_graph(log, threshold=0.05):
    DG = nx.DiGraph()
    matrix = markov_chain_adjacency_matrix(log)
    number_for_event = ranking_dict(log)
    event_for_number = dict((y, x) for x, y in number_for_event.items())
    for row in range(len(matrix)):
        for event in range(len(matrix[row])):
            if matrix[row][event] > threshold:
                DG.add_edge(event_for_number[row], event_for_number[event], weight='{:.2f}'.format(matrix[row][event]))
    return DG


# Create undirected weighted graph
def create_undirected_weighted_graph(log, threshold=0.05):
    DG = nx.Graph()
    matrix = markov_chain_adjacency_matrix(log)
    number_for_event = ranking_dict(log)
    event_for_number = dict((y, x) for x, y in number_for_event.items())
    for row in range(len(matrix)):
        for event in range(len(matrix[row])):
            if matrix[row][event] > threshold:
                DG.add_edge(event_for_number[row], event_for_number[event], weight='{:.2f}'.format(matrix[row][event]))
                DG.add_edge(event_for_number[event], event_for_number[row], weight='{:.2f}'.format(matrix[row][event]))
    return DG
