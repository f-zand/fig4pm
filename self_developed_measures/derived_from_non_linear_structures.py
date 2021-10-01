# METHODS IMPLEMENTING SELF-DEVELOPED MEASURES
# DERIVED FROM NON-LINEAR STRUCTURES OF THE EVENT LOG
from graph_creation import create_undirected_graph, create_directed_graph
import networkx as nx
import numpy as np

# 1. Number of graph communities using greedy modularity
def number_of_graph_communities(log):
    graph = create_undirected_graph(log)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    return len(communities)

# 2. Cut vertex outgoing degree
def maximum_cut_vertex_outgoing_degree(log):
    undirected_graph = create_undirected_graph(log)
    directed_graph = create_directed_graph(log)
    cuts = list(nx.articulation_points(undirected_graph))
    max_outgoing_degree = 0
    for i in cuts:
        outgoing_degree = directed_graph.out_degree(i)
        if outgoing_degree > max_outgoing_degree:
            max_outgoing_degree = outgoing_degree
    return max_outgoing_degree

# 3. Cut vertex outgoing degree
def cut_vertex_independent_path(log):
    undirected_graph = create_undirected_graph(log)
    directed_graph = create_directed_graph(log)
    cuts = list(nx.articulation_points(undirected_graph))
    max_disjoint_paths = 0
    for i in cuts:
        for j in cuts:
            if i is not j:
                try:
                    number_of_paths = len(list(nx.node_disjoint_paths(directed_graph, i, j)))
                except nx.exception.NetworkXNoPath:
                    number_of_paths = 0
                if number_of_paths > max_disjoint_paths:
                    max_disjoint_paths = number_of_paths
    return max_disjoint_paths


# 4. Simple path minimum jaccard similarity
def simple_path_minimum_jaccard_similarity(log, threshold=0.05):
    from pm4py.statistics.start_activities.log.get import get_start_activities
    from pm4py.statistics.end_activities.log.get import get_end_activities
    from general_methods import jaccard_similarity
    from tqdm import tqdm
    from graph_creation import create_directed_weighted_graph

    # Retrieve start and end events
    start_events = [*get_start_activities(log)]
    end_events = [*get_end_activities(log)]
    start_events_count = list(get_start_activities(log).values())
    end_events_count = list(get_end_activities(log).values())

    # Only keep frequent events to reduce computation time
    start_events_to_keep = []
    for i in range(len(start_events)):
        if start_events_count[i] / max(start_events_count) > threshold:
            start_events_to_keep.append(start_events[i])
    end_events_to_keep = []
    for i in range(len(end_events)):
        if end_events_count[i] / max(end_events_count) > threshold:
            end_events_to_keep.append(end_events[i])

    graph = create_directed_weighted_graph(log, threshold=0.05)
    simple_paths = []

    # create list of simple paths
    for i in start_events_to_keep:
        for j in end_events_to_keep:
            simple_paths.append(list(nx.all_simple_paths(graph, i, j)))
    simple_paths = np.asarray(list(elem for sub in simple_paths for elem in sub))
    similarity = []

    # Set up progress bar
    no_trace = len(simple_paths)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Jaccard distance all simple paths, completed :: ")

    # create list containing similarities between all simple paths
    for i in simple_paths:
        progress.update()
        for j in simple_paths:
            similarity.append(jaccard_similarity(i, j))

    # Close progress bar and return
    progress.close()
    del progress
    return min(similarity)


# 5. Syntactic node similarity
def syntactic_node_similarity(log):
    from editdistance import distance
    from general_methods import event_names

    events = event_names(log)
    counter = 0
    for i in events:
        for j in events:
            sim = 1 - (distance(i, j) / max(len(i), len(j)))
            if sim >= 0.5:
                counter += 1
    return (counter - len(events)) / (len(events) * (len(events) - 1) - len(events))
