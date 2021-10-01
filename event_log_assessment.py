###############################################################################
'''Necessery liberaries'''
###############################################################################
import pandas as pd
import pm4py
from time import time
from pm4py.objects.conversion.log import converter as log_converter
from measures_extracted_from_literature.derived_from_linear_structures import *
from measures_extracted_from_literature.derived_from_non_linear_structures import *
from self_developed_measures.derived_from_linear_structures import *
from self_developed_measures.derived_from_non_linear_structures import *

###############################################################################
'''Event log assessment function'''
###############################################################################
def event_log_assessment(log):

    t0 = time()

    # 1. derived from linear structures
    LS = {}

    # 1.1. Returns the number of events in the event log (ne)
    LS['total number of events'] = total_number_of_events(log)


    # 1.2. Returns the number of event classes in the event log (nec)
    LS['Total number of event classes'] = total_number_of_event_classes(log)


    # 1.3. Returns the number of traces in the event log (nt)
    LS['Total number of traces'] = total_number_of_traces(log)


    # 1.4. Returns the number of trace classes (i.e. variants) in the event log (ntc)
    LS['Total number of trace classes'] = total_number_of_trace_classes(log)


    # 1.5. Returns the average trace length (atl)
    #      (i.e. total number of events in the log divided by total number of cases (traces) in the log)
    LS['Average trace length'] = average_trace_length(log)


    # 1.6. Returns length of shortest trace (mitl)
    LS['Minimum trace length'] = minimum_trace_length(log)


    # 1.7. Returns length of longest trace (matl)
    LS['Maximum trace length'] = maximum_trace_length(log)


    # 1.8. Average trace size (i.e. number of event classes per case (trace)) (ats)
    LS['Average trace size'] = average_trace_size(log)
    ### have to more think 


    # 1.9. Absolute number of start events (nsec)
    LS['Number of distinct start events'] = number_of_distinct_start_events(log)


    # 1.10. Absolute number of end events (ntec)
    LS['Number of distinct end events'] = number_of_distinct_end_events(log)


    # 1.11. Absolute number of traces including a self-loop of at least length 1 (ntsl)
    LS['Absolute number of traces with self loop'] = absolute_number_of_traces_with_self_loop(log)


    # 1.12. Absolute number of traces including a repetition (excluding loops) (ntr)
    LS['Absoulute number of traces with repetition'] = absoulute_number_of_traces_with_repetition(log)


    # 1.13. Relative number of start activities (rnsec)
    LS['Relative number of distinct start events'] = relative_number_of_distinct_start_events(log)


    # 1.14. Relative number of end activities (rntec)
    LS['Rlative number of distinct end events'] = relative_number_of_distinct_end_events(log)


    # 1.15. Relative number of traces including a self-loop of at least length 1 (rntsl)
    LS['Relative number of traces with self loop'] = relative_number_of_traces_with_self_loop(log)


    # 1.16. Relative number of traces including a repetition (excluding loops) (rntr)
    LS['Relative number of traces with repetition'] = relative_number_of_traces_with_repetition(log)


    # 1.17. Average number of self-loops per trace (anslt)
    LS['Average number of self loops per trace'] = average_number_of_self_loops_per_trace(log)


    # 1.18. Maximum number of self-loops per trace (manslt)
    LS['Maximum number of self loops per trace'] = maximum_number_of_self_loops_per_trace(log)


    # 1.19. Average size of self-loops per trace (only accounting for traces that contain a self-loop) (asslt)
    LS['Average size of self loops per trace'] = average_size_of_self_loops_per_trace(log)


    # 1.20. Maximum size of self-loops for any trace (masslt)
    LS['Maximum size of self loops per trace'] = maximum_size_of_self_loops_per_trace(log)


    # 1.21. Absolute number of distinct traces per 100 traces (tcpht)
    LS['Number of distinct traces per hundred traces'] = number_of_distinct_traces_per_hundred_traces(log)


    # 1.22. Absolute trace coverage: 80 percent level (tco)
    LS['Absolute trace coverage'] = absolute_trace_coverage(log)


    # 1.23. Absolute trace coverage: 80 percent level (rtco)
    LS['Relative trace coverage'] = relative_trace_coverage(log)


    # 1.24. Event density (i.e. average trace size / average trace length) (edn)
    LS['Event density'] = event_density(log)


    # 1.25. Traces heterogeneity rate (i.e. ln(variant_count) / ln(case_count)) (thr)
    LS['Traces heterogeneity rate'] = traces_heterogeneity_rate(log)


    # 1.26. Trace similarity rate (tsr)
    LS['Trace similarity rate'] = trace_similarity_rate(log)


    # 1.27. Complexity factor (cf)
    LS['Complexity factor'] = complexity_factor(log)


    # 1.28. Simple trace diversity (i.e. 1 - (average trace size / total number of activities)) (std)
    LS['Simple trace diversity'] = simple_trace_diversity(log)


    # 1.29. Advanced trace diversity (i.e. weighted levenshtein distance between all traces) (atd)
    LS['Advanced trace diversity'] = advanced_trace_diversity(log)


    # 1.30. Trace entropy (tentr)
    LS['Trace entropy'] = trace_entropy(log)


    # 1.31. Prefix entropy (flattened) (prentr)
    LS['Prefix entropy'] = prefix_entropy(log)


    # 1.32. All-block entropy (flattened) (abentr)
    LS['All block entropy'] = all_block_entropy(log)



    # 2. derived from non-linear structures
    NLS = {}


    # 2.1. Number of nodes in the graph (i.e. events in the log) (N)
    NLS['Number of nodes'] = number_of_nodes(log)


    # 2.2. Number of arcs in the graph (i.e. transitions between events in the log) (A)
    NLS['Number of arcs'] = number_of_arcs(log)


    # 2.3. Coefficient of network connectivity / complexity (i.e. number of arcs / number of nodes) (gcnc)
    NLS['Coefficient of network connectivity'] = coefficient_of_network_connectivity(log)


    # 2.4. Average node degree (i.e. (2 x number of arcs) / number of nodes) (gand)
    NLS['Average node degree'] = average_node_degree(log)


    # 2.5. Maximum node degree (gmnd)
    NLS['Maximum node degree'] = maximum_node_degree(log)


    # 2.6. Density (i.e. A / (N x (N-1)) (gdn)
    NLS['Density'] = density(log)


    # 2.7. Structure (i.e. 1 - (A / (N^2))) (gst)
    NLS['Structure'] = structure(log)


    # 2.8. Absolute cyclomatic number (i.e. A - N + 1) (gcn)
    NLS['Cyclomatic number'] = cyclomatic_number(log)


    # 2.9. Graph diameter, i.e. longest path through the process without accounting for cycles (gdm)
    NLS['Graph diameter'] = graph_diameter(log)


    # 2.10. Absolute number of cut vertices, i.e. articulation points,
    #     that separate the graph into several components when removed (gcv)
    NLS['Number of cut vertices'] = number_of_cut_vertices(log)


    # 2.11. Separability ratio (gsepr)
    NLS['Separability ratio'] = separability_ratio(log)


    # 2.12. Sequentiality ratio (gseqr)
    NLS['Sequentiality ratio'] = sequentiality_ratio(log)


    # 2.13. Cyclicitly (gcy)
    NLS['Cyclicity'] = cyclicity(log)


    # 2.14. Affinity (gaf)
    NLS['Affinity'] = affinity(log)


    # 2.15. Simple Path Process Complexity (gspc)
    NLS['simple path complexity'] = simple_path_complexity(log)

    # 3. derived from self linear structures
    SLS = {}

    # 3.1. Outlier evaluation of start event frequencies
    SLS['start event frequency evaluation'] = start_event_frequency_evaluation(log, 'highest_occurrence', 0.05)

    # 3.2. Outlier evaluation of end event frequencies
    SLS['end event frequency evaluation'] = end_event_frequency_evaluation(log, 'highest_occurrence', 0.05)

    # 3.3. Outlier evaluation of event frequencies
    SLS['event frequency evaluation'] = event_frequency_evaluation(log, 'highest_occurrence', 0.05)

    # 3.4. Outlier evaluation of trace frequencies
    SLS['trace frequency evaluation'] = trace_frequency_evaluation(log, 'highest_occurrence', 0.05)

    # 3.5. Outlier evaluation of event dependency
    SLS['event dependency evaluation'] = event_dependency_evaluation(log)

    # 3.6. Outlier evaluation of trace length
    SLS['trace length evaluation'] = trace_length_evaluation(log)

    # 3.7. Absolute number of outlying traces detected via unsupervised outlier detection algorithm
    SLS['number of outlying traces'] = number_of_outlying_traces(log)

    # 3.8 Relative number of outlying traces detected via unsupervised outlier detection algorithm
    SLS['relative number of outlying traces'] = relative_number_of_outlying_traces(log)

    # 3.9. Event profile average euclidean distance
    SLS['event profile average euclidean distance'] = event_profile_average_euclidean_distance(log)

    # 3.10. Event profile average cosine distance
    SLS['event profile average cosine similarity'] = event_profile_average_cosine_similarity(log)

    # 3.11. Transition profile average euclidean distance
    SLS['transition profile average euclidean distancedistance'] = transition_profile_average_euclidean_distance(log)

    # 3.12. Transition profile average cosine distance
    SLS['transition profile average cosine similarity'] = transition_profile_average_cosine_similarity(log)

    # 3.13. Event profile maximum cosine distance
    SLS['event profile minimum cosine similarity'] = event_profile_minimum_cosine_similarity(log)

    # 3.14. Transition profile maximum cosine distance
    SLS['transition profile minimum cosine similarity'] = transition_profile_minimum_cosine_similarity(log)

    # 3.15. Average spatial proximity
    SLS['average spatial proximity'] = average_spatial_proximity(log)

    # 3.16. Spatial proximity connectedness
    SLS['spatial proximity connectedness'] = spatial_proximity_connectedness(log)

    # 3.17. Spatial proximity abstraction evaluation
    SLS['spatial proximity abstraction evaluation'] = spatial_proximity_abstraction_evaluation(log)

    # 3.18 Event dependency abstraction evaluation
    SLS['event dependency abstraction evaluation'] = event_dependency_abstraction_evaluation(log)

    # 3.19. Triple abstraction evaluation
    SLS['triple abstraction evaluation'] = triple_abstraction_evaluation(log)

    # 3.20. Event class triple abstraction evaluation
    SLS['event class triple abstraction evaluation'] = event_class_triple_abstraction_evaluation(log)

    # 4. derived from self non-linear structures
    SNLS = {}

    # 4.1. Number of graph communities using greedy modularity
    SNLS['number of graph communities'] = number_of_graph_communities(log)

    # 4.2. Cut vertex outgoing degree
    SNLS['maximum cut vertex outgoing degree'] = maximum_cut_vertex_outgoing_degree(log)

    # 4.3. Cut vertex outgoing degree
    SNLS['cut vertex independent path'] = cut_vertex_independent_path(log)

    # 4.4. Simple path minimum jaccard similarity
    SNLS['simple path minimum jaccard similarity'] = simple_path_minimum_jaccard_similarity(log)

    # 4.5. Syntactic node similarity
    SNLS['syntactic node similarity'] = syntactic_node_similarity(log)


    print(pd.DataFrame(list(LS.items()), columns = ['*linear structure*', '*value*']))
    print()
    print()
    print(pd.DataFrame(list(NLS.items()), columns = ['*non-linear structure*', '*value*']))
    print()
    print()
    print(pd.DataFrame(list(SLS.items()), columns = ['*self linear structure*', '*value*']))
    print()
    print()
    print(pd.DataFrame(list(SNLS.items()), columns = ['*self non-linear structure*', '*value*']))

    t1 = time()
    print()
    print('-------------------------------------------')
    print('runing time:', t1 - t0)
    print('-------------------------------------------')
    print()

###############################################################################
'''Loading CSV event log file'''
###############################################################################
event_log = pm4py.format_dataframe(pd.read_csv('EventLog\complex-eventlog.csv', sep=';'),
                                    case_id='ValueAddedServiceID+BatchID',
                                    activity_key='Activity_Description',
                                    timestamp_key='StartTimeStamp')

event_log = log_converter.apply(event_log)

print()
print('-------------------------------------------')
print('Event Log Imported!')
print('-------------------------------------------')

###############################################################################
''''Runing Program'''
###############################################################################
event_log_assessment(event_log)

###############################################################################