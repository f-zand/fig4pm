# METHODS IMPLEMENTING MEASURES EXTRACTED FROM THE LITERATURE
# DERIVED FROM NON-LINEAR STRUCTURES OF THE EVENT LOG


# Number of nodes in the graph (i.e. events in the log) (N)
def number_of_nodes(log):
    from graph_creation import create_directed_graph
    import networkx as nx
    graph = create_directed_graph(log)
    return len(nx.nodes(graph))


# Number of arcs in the graph (i.e. transitions between events in the log) (A)
def number_of_arcs(log):
    from graph_creation import create_directed_graph
    import networkx as nx
    graph = create_directed_graph(log)
    return len(nx.edges(graph))


# Coefficient of network connectivity / complexity (i.e. number of arcs / number of nodes) (gcnc)
def coefficient_of_network_connectivity(log):
    return number_of_arcs(log) / number_of_nodes(log)


# Average node degree (i.e. (2 x number of arcs) / number of nodes) (gand)
def average_node_degree(log):
    return (2 * number_of_arcs(log)) / number_of_nodes(log)


# Maximum node degree (gmnd)
def maximum_node_degree(log):
    from graph_creation import create_directed_graph
    import networkx as nx
    graph = create_directed_graph(log)
    degrees = list(i[1] for i in nx.degree(graph))
    return max(degrees)


# Density (i.e. A / (N x (N-1)) (gdn)
def density(log):
    from graph_creation import create_directed_graph
    import networkx as nx
    graph = create_directed_graph(log)
    return nx.density(graph)


# Structure (i.e. 1 - (A / (N^2))) (gst)
def structure(log):
    return 1 - (number_of_arcs(log) / (number_of_nodes(log)**2))


# Absolute cyclomatic number (i.e. A - N + 1) (gcn)
def cyclomatic_number(log):
    cn = number_of_arcs(log) - number_of_nodes(log) + 1
    return cn


# Graph diameter, i.e. longest path through the process without accounting for cycles (gdm)
def graph_diameter(log, threshold=0.05):
    import numpy as np
    from pm4py.statistics.start_activities.log.get import get_start_activities
    from pm4py.statistics.end_activities.log.get import get_end_activities
    from graph_creation import create_directed_weighted_graph
    import networkx as nx
    # create list of start and end events as well as graph
    start_events = [*get_start_activities(log)]
    end_events = [*get_end_activities(log)]
    start_events_count = list(get_start_activities(log).values())
    end_events_count = list(get_end_activities(log).values())

    # in order to enhance computation time - only select frequent start and end events
    start_events_to_keep = []
    for i in range(len(start_events)):
        if start_events_count[i] / max(start_events_count) > threshold:
            start_events_to_keep.append(start_events[i])
    end_events_to_keep = []
    for i in range(len(end_events)):
        if end_events_count[i] / max(end_events_count) > threshold:
            end_events_to_keep.append(end_events[i])

    graph = create_directed_weighted_graph(log, threshold=0.05)

    # retrieve all simple paths, i.e. paths without cycles
    simple_path_list = []
    for i in start_events_to_keep:
        for j in end_events_to_keep:
            simple_path_list.append(list(nx.all_simple_paths(graph, i, j)))
    simple_paths = np.asarray(list(elem for sub in simple_path_list for elem in sub))
    # retrieve all simple path lengths
    simple_path_length = list(len(i) for i in list(simple_paths))
    return max(simple_path_length)


# Absolute number of cut vertices, i.e. articulation points,
# that separate the graph into several components when removed (gcv)
def number_of_cut_vertices(log):
    from graph_creation import create_undirected_graph
    import networkx as nx
    graph = create_undirected_graph(log)
    return len(list(nx.articulation_points(graph)))


# Separability ratio (gsepr)
def separability_ratio(log):
    return number_of_cut_vertices(log) / number_of_nodes(log)


# Sequentiality ratio (gseqr)
def sequentiality_ratio(log):
    from graph_creation import create_directed_graph
    # retrieve graph and list of in- and out-degrees of each node
    graph = create_directed_graph(log)
    in_degree_list = graph.in_degree
    out_degree_list = graph.out_degree
    # create list of nodes that only have one incoming and one outgoing edge, i.e. non-connector nodes
    in_degree_events = [i[0] for i in in_degree_list if i[1] == 1 or i[1] == 0]
    out_degree_events = [i[0] for i in out_degree_list if i[1] == 1 or i[1] == 0]
    non_connector_nodes = list(set(in_degree_events).intersection(out_degree_events))
    # count edges between non-connector nodes
    non_connector_edges = 0
    for i in non_connector_nodes:
        for j in non_connector_nodes:
            if graph.has_edge(i, j):
                non_connector_edges += 1
    return non_connector_edges / number_of_arcs(log)


# Cyclicitly (gcy)
def cyclicity(log):
    import networkx as nx
    from graph_creation import create_directed_graph
    # retrieve cycles and set of nodes contained in the cycles
    graph = create_directed_graph(log)
    cycles = list(nx.simple_cycles(graph))
    cycle_nodes = set()
    for i in cycles:
        if len(i) > 1:
            cycle_nodes = cycle_nodes.union(set(i))
    return len(list(cycle_nodes)) / number_of_nodes(log)


# Affinity (gaf)
def affinity(log):
    from general_methods import case_list, jaccard_similarity
    # create transition representation of all traces in the log
    traces = case_list(log)
    trace_transition_representation = []
    for trace in traces:
        transitions = []
        for i in range(len(trace) - 1):
            transitions.append((trace[i], trace[i + 1]))
        trace_transition_representation.append(transitions)
    # measure average affinity, i.e. relative overlap between all traces in the log
    total_overlap = 0
    for reference_trace in trace_transition_representation:
        for compared_trace in trace_transition_representation:
            total_overlap += jaccard_similarity(reference_trace, compared_trace)
    return total_overlap / (len(traces) * (len(traces) - 1))


# Simple Path Process Complexity (gspc)
def simple_path_complexity(log, threshold=0.05):
    from pm4py.statistics.start_activities.log.get import get_start_activities
    from pm4py.statistics.end_activities.log.get import get_end_activities
    from graph_creation import create_directed_graph
    import networkx as nx

    # create list of start and end events as well as graph
    start_events = [*get_start_activities(log)]
    end_events = [*get_end_activities(log)]
    start_events_count = list(get_start_activities(log).values())
    end_events_count = list(get_end_activities(log).values())

    # in order to enhance computation time - only select frequent start and end events
    start_events_to_keep = []
    for i in range(len(start_events)):
        if start_events_count[i] / max(start_events_count) > threshold:
            start_events_to_keep.append(start_events[i])
    end_events_to_keep = []
    for i in range(len(end_events)):
        if end_events_count[i] / max(end_events_count) > threshold:
            end_events_to_keep.append(end_events[i])

    graph = create_directed_graph(log)

    # count number of simple paths
    simple_path_count = 0
    for i in start_events:
        for j in end_events:
            simple_path_count += len(list(nx.all_simple_paths(graph, i, j)))
    return simple_path_count
