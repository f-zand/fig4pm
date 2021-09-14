# GENERAL METHODS USED THROUGHOUT THE EVENT LOG ASSESSMENT TOOL
import numpy as np
from pm4py.objects.dfg.retrieval.log import native
from pm4py.statistics.traces.log import case_statistics

# Returns the names of all events in the log
def event_names(log):
    from pm4py.algo.filtering.log.attributes import attributes_filter as log_attributes_filter

    events = log_attributes_filter.get_attribute_values(log, "concept:name")
    event_list = [*events]
    return event_list


# Returns a list including all cases represented as a sequence of their event names
def case_list(log):
    log_rep = []
    for case in log:
        case_rep = []
        for event in case:
            case_rep.append(event["concept:name"])
        log_rep.append(case_rep)
    return np.asarray(log_rep)


# Returns a dictionary including the variants of the log as the key and their count as a value
def variant_dict(log):
    var_dict = sorted(case_statistics.get_variant_statistics(log), key=lambda x: x['count'], reverse=True)
    return var_dict


# Returns a list including all variants represented as a sequence of their event names
def variant_list(log):
    log_rep = []
    for case in log:
        case_rep = []
        for event in case:
            case_rep.append(event["concept:name"])
        if case_rep not in log_rep:
            log_rep.append(case_rep)
    return np.asarray(log_rep)


# Returns a list including all frequencies of variants corresponding to the order in variant_list
def variant_count_list(log):
    variants_dict = sorted(case_statistics.get_variant_statistics(log), key=lambda x: x['count'], reverse=True)
    var_count_list = []
    for item in variants_dict:
        var_count_list.append(item["count"])
    return np.asarray(var_count_list)


# Returns a dictionary that distributes a distinct number to every event occurring in the event log
def ranking_dict(log):
    events = sorted(event_names(log))
    rank_dict = {b: a for a, b in enumerate(events)}
    return rank_dict


# Creates an adjacency matrix for the complete log, on which a directed graph can be based
def adjacency_matrix_directed(log):
    event_ranking = ranking_dict(log)
    connections = [list(i) for i in [*(native(log))]]

    # Bring connections in rank format
    for connection in range(len(connections)):
        for elem in range(len(connections[connection])):
            connections[connection][elem] = event_ranking[connections[connection][elem]]

    # Create initial matrix
    matrix_shape = len(event_names(log))
    adjac_matrix = np.zeros(shape=(matrix_shape, matrix_shape))

    # Fill matrix based on connections
    for (i, j) in connections:
        adjac_matrix[i][j] += 1
    return np.asarray(adjac_matrix)


# Creates an adjacency matrix for the complete log, on which an undirected graph can be based
def adjacency_matrix_undirected(log):

    event_ranking = ranking_dict(log)
    connections = [list(i) for i in [*(native(log))]]

    # Bring connections in rank format
    for connection in range(len(connections)):
        for elem in range(len(connections[connection])):
            connections[connection][elem] = event_ranking[connections[connection][elem]]

    # Create initial matrix
    matrix_shape = len(event_names(log))
    adjac_matrix = np.zeros(shape=(matrix_shape, matrix_shape))

    # Fill matrix based on connections
    for (i, j) in connections:
        adjac_matrix[i][j] += 1
        adjac_matrix[j][i] += 1
    return np.asarray(adjac_matrix)


# Create markov chain adjacency matrix for the creation of a weighted graph
def markov_chain_adjacency_matrix(log):
    # Create relevant dictionaries
    number_for_event = ranking_dict(log)
    event_for_number = dict((y, x) for x, y in number_for_event.items())
    event_connection_count = native(log)
    event_connections = [list(i) for i in [*(native(log))]]

    # Bring connections in rank format
    for connection in range(len(event_connections)):
        for elem in range(len(event_connections[connection])):
            event_connections[connection][elem] = number_for_event[event_connections[connection][elem]]

    # Create initial matrix
    matrix_shape = len(event_names(log))
    adjac_matrix = np.zeros(shape=(matrix_shape, matrix_shape))

    # Fill matrix based on connections
    for (i, j) in event_connections:
        adjac_matrix[i][j] += event_connection_count[(event_for_number[i], event_for_number[j])]
    for row in adjac_matrix:
        n = sum(row)
        if n > 0:
            row[:] = [(f / n) for f in row]
    return np.asarray(adjac_matrix)


# Jaccard similarity computation
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# FOUNDATION FOR METHODS EVALUATING SELF-LOOPS AND REPETITIONS IN THE EVENT LOG

# Log-based self-loop list: list that shows the number and size of self-loops for every trace
def self_loop_per_trace_overview(log):
    self_loop_overview_list = []
    # create trace representation
    for case in log:
        trace = []
        for event in case:
            trace.append(event["concept:name"])
        # create list containing size of loops in trace
        self_loop_size_list = []
        i = 0
        while i < (len(trace) - 1):
            self_loop_size = 0
            # if self-loop is detected, measure size of it
            if trace[i] == trace[i + 1]:
                for k in range(i + 1, len(trace), 1):
                    if trace[i] == trace[k]:
                        self_loop_size += 1
            if self_loop_size > 0:
                self_loop_size_list.append(self_loop_size)
                i += self_loop_size
            else:
                i += 1
        self_loop_overview_list.append(self_loop_size_list)
    return self_loop_overview_list


# Log-based repetition list: list that shows the number and size of repetitions for every trace
def repetition_per_trace_overview(log):
    repetition_overview_list = []
    # create trace representation
    for case in log:
        trace = []
        for event in case:
            trace.append(event["concept:name"])
        # create window to detect repetitions
        window = []
        repetition_size_list = []
        repetition_size = 0
        # append events to window unless an event is added for a second time
        for i in trace:
            if i not in window:
                window.append(i)
            else:
                # check if repetition is not a self-loop
                position = len(window) - 1 - window[::-1].index(i)
                if position == (len(window) - 1):
                    window.append(i)
                else:
                    # calculate repetition size and delete repetition from window
                    repetition_size += len(window[position: (len(window) + 1)])
                    repetition_size_list.append(repetition_size)
                    del window[position: (len(window) + 1)]
                    window.append(i)
        repetition_overview_list.append(repetition_size_list)
    return repetition_overview_list

