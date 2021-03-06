# METHOD USED FOR EVALUATING THE MEASURES PROPOSED THROUGHOUT THE THESIS
# STRUCTURED ALONG THE RESPECTIVE SECTIONS

# Import relevant modules
import xlsxwriter
from time import time
from datetime import timedelta
from measures_extracted_from_literature.derived_from_linear_structures import *
from measures_extracted_from_literature.derived_from_non_linear_structures import *
from self_developed_measures.derived_from_linear_structures import *
from self_developed_measures.derived_from_non_linear_structures import *


# Measure evaluation
def measure_evaluation(log):

    print()
    print('Log Imported!')
    print()
    t1 = time()

    if total_number_of_trace_classes(log) == 1:  # assessment does not work for event logs that only contain one variant
        raise Exception('Evaluation not possible - Log only contains one variant!')

    if minimum_trace_length(log) == 1:  # feature profile creation does not work for traces that only contain one event
        raise Exception('Evaluation not possible - Log contains at least one variant only including one event - please remove those trace/s!')

    # create workbook
    workbook = xlsxwriter.Workbook('01_Measure_Evaluation.xlsx')
    bold = workbook.add_format({'bold': True})
    underline = workbook.add_format({'underline': True})

    worksheet = workbook.add_worksheet('Evaluation')
    worksheet.set_column(0, 0, 2)
    worksheet.set_column(1, 1, 90)
    worksheet.hide_gridlines(2)

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(1, 1, '-------------------------------------------------------------------', bold)
    worksheet.write(2, 1, 'Measures extracted from the literature:', bold)
    worksheet.write(3, 1, '-------------------------------------------------------------------', bold)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(5, 1, 'Derived from linear structures (i.e. matrix, vector, scalar)', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(7, 1, '- Total number of events:')
    worksheet.write(7, 2, str(total_number_of_events(log)))
    worksheet.write(9, 1, '- Total number of event classes:')
    worksheet.write(9, 2, str(total_number_of_event_classes(log)))
    worksheet.write(11, 1, '- Total number of traces:')
    worksheet.write(11, 2, str(total_number_of_traces(log)))
    worksheet.write(13, 1, '- Total number of trace classes (i.e. variants):')
    worksheet.write(13, 2, str(total_number_of_trace_classes(log)))
    worksheet.write(15, 1, '- Average trace length:')
    worksheet.write(15, 2, str('{:.2f}'.format(average_trace_length(log))))
    worksheet.write(17, 1, '- Minimum trace length:')
    worksheet.write(17, 2, str(minimum_trace_length(log)))
    worksheet.write(19, 1, '- Maximum trace length:')
    worksheet.write(19, 2, str(maximum_trace_length(log)))
    worksheet.write(21, 1, '- Average trace size:')
    worksheet.write(21, 2, str('{:.2f}'.format(average_trace_size(log))))
    worksheet.write(23, 1, '- Absolute number of distinct start events:')
    worksheet.write(23, 2, str(number_of_distinct_start_events(log)))
    worksheet.write(25, 1, '- Absolute number of distinct end events:')
    worksheet.write(25, 2, str(number_of_distinct_end_events(log)))
    worksheet.write(27, 1, '- Absolute number of traces with a self-loop:')
    worksheet.write(27, 2, str(absolute_number_of_traces_with_self_loop(log)))
    worksheet.write(29, 1, '- Absolute number of traces with a repetition:')
    worksheet.write(29, 2, str(absoulute_number_of_traces_with_repetition(log)))
    worksheet.write(31, 1, '- Relative number of distinct start events:')
    worksheet.write(31, 2, str('{:.2f}'.format(relative_number_of_distinct_start_events(log))))
    worksheet.write(33, 1, '- Relative number of distinct end events:')
    worksheet.write(33, 2, str('{:.2f}'.format(relative_number_of_distinct_end_events(log))))
    worksheet.write(35, 1, '- Relative number of traces with a self-loop:')
    worksheet.write(35, 2, str('{:.2f}'.format(relative_number_of_traces_with_self_loop(log))))
    worksheet.write(37, 1, '- Relative number of traces with a repetition:')
    worksheet.write(37, 2, str('{:.2f}'.format(relative_number_of_traces_with_repetition(log))))
    worksheet.write(39, 1, '- Average number of self-loops per trace:')
    worksheet.write(39, 2, str('{:.2f}'.format(average_number_of_self_loops_per_trace(log))))
    worksheet.write(41, 1, '- Maximum number of self-loops per trace:')
    worksheet.write(41, 2, str('{:.2f}'.format(maximum_number_of_self_loops_per_trace(log))))
    worksheet.write(43, 1, '- Average size of self-loops per trace:')
    worksheet.write(43, 2, str('{:.2f}'.format(average_size_of_self_loops_per_trace(log))))
    worksheet.write(45, 1, '- Maximum size of self-loops per trace:')
    worksheet.write(45, 2, str('{:.2f}'.format(maximum_size_of_self_loops_per_trace(log))))
    worksheet.write(47, 1, '- Number of distinct traces per 100 traces:')
    worksheet.write(47, 2, str('{:.2f}'.format(number_of_distinct_traces_per_hundred_traces(log))))
    worksheet.write(49, 1, '- Absolute trace coverage:')
    worksheet.write(49, 2, str(absolute_trace_coverage(log)))
    worksheet.write(51, 1, '- Relative trace coverage:')
    worksheet.write(51, 2, str('{:.2f}'.format(relative_trace_coverage(log))))
    worksheet.write(53, 1, '- Event density:')
    worksheet.write(53, 2, str('{:.2f}'.format(event_density(log))))
    worksheet.write(55, 1, '- Trace heterogeneity rate (log normalized):')
    worksheet.write(55, 2, str('{:.2f}'.format(traces_heterogeneity_rate(log))))
    worksheet.write(57, 1, '- Trace similarity rate:')
    worksheet.write(57, 2, str('{:.2f}'.format(trace_similarity_rate(log))))
    worksheet.write(59, 1, '- Complexity factor:')
    worksheet.write(59, 2, str('{:.2f}'.format(complexity_factor(log))))
    worksheet.write(60, 1, '  Complexity Factor Ranking:')
    worksheet.write(61, 1, '  0 - 30: Simple')
    worksheet.write(62, 1, '  30 - 53: Somewhat Complex')
    worksheet.write(63, 1, '  53 - 75: Complex')
    worksheet.write(64, 1, '  75 - 95: Very Complex')
    worksheet.write(65, 1, '  > 95: Highly Complex')
    worksheet.write(67, 1, '- Simple trace diversity:')
    worksheet.write(67, 2, str('{:.2f}'.format(simple_trace_diversity(log))))
    advanced_tr_div = advanced_trace_diversity(log)
    worksheet.write(69, 1, '- Advanced trace diversity (full steps):')
    worksheet.write(69, 2, str('{:.2f}'.format(advanced_tr_div * average_trace_length(log))))
    worksheet.write(71, 1, '- Advanced trace diversity (relative):')
    worksheet.write(71, 2, str('{:.2f}'.format(advanced_tr_div)))
    worksheet.write(73, 1, '- Trace entropy:')
    worksheet.write(73, 2, str('{:.2f}'.format(trace_entropy(log))))
    worksheet.write(75, 1, '- Prefix entropy (flattened):')
    worksheet.write(75, 2, str('{:.2f}'.format(prefix_entropy(log))))
    worksheet.write(77, 1, '- All-block entropy (flattened):')
    worksheet.write(77, 2, str('{:.2f}'.format(all_block_entropy(log))))

    print('Measures extracted from the literature - derived from linear structures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(79, 1, 'Derived from non-linear features (i.e. graph)', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(81, 1, '- Number of nodes:')
    worksheet.write(81, 2, str(number_of_nodes(log)))
    worksheet.write(83, 1, '- Number of arcs:')
    worksheet.write(83, 2, str(number_of_arcs(log)))
    worksheet.write(85, 1, '- Coefficient of network connectivity:')
    worksheet.write(85, 2, str('{:.2f}'.format(coefficient_of_network_connectivity(log))))
    worksheet.write(87, 1, '- Average node degree:')
    worksheet.write(87, 2, str('{:.2f}'.format(average_node_degree(log))))
    worksheet.write(89, 1, '- Maximum node degree:')
    worksheet.write(89, 2, str('{:.2f}'.format(maximum_node_degree(log))))
    worksheet.write(91, 1, '- Graph Density:')
    worksheet.write(91, 2, str('{:.2f}'.format(density(log))))
    worksheet.write(93, 1, '- Structure:')
    worksheet.write(93, 2, str('{:.2f}'.format(structure(log))))
    worksheet.write(95, 1, '- Cyclomatic number of the graph:')
    worksheet.write(95, 2, str('{:.2f}'.format(cyclomatic_number(log))))
    worksheet.write(97, 1, '- Graph diameter:')
    worksheet.write(97, 2, str('{:.2f}'.format(graph_diameter(log))))
    worksheet.write(99, 1, '- Number of cut-vertices:')
    worksheet.write(99, 2, str('{:.2f}'.format(number_of_cut_vertices(log))))
    worksheet.write(101, 1, '- Separability ratio:')
    worksheet.write(101, 2, str('{:.2f}'.format(separability_ratio(log))))
    worksheet.write(103, 1, '- Sequentiality ratio:')
    worksheet.write(103, 2, str('{:.2f}'.format(sequentiality_ratio(log))))
    worksheet.write(105, 1, '- Cyclicity:')
    worksheet.write(105, 2, str('{:.2f}'.format(cyclicity(log))))
    worksheet.write(107, 1, '- Affinity:')
    worksheet.write(107, 2, str('{:.2f}'.format(affinity(log))))
    worksheet.write(109, 1, '- Simple Path Complexity:')
    worksheet.write(109, 2, str(simple_path_complexity(log)))

    print('Measures extracted from the literature - derived from non-linear structures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(111, 1, '-------------------------------------------------------------------', bold)
    worksheet.write(112, 1, 'Self-developed measures:', bold)
    worksheet.write(113, 1, '-------------------------------------------------------------------', bold)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(115, 1, 'Derived from linear structures (i.e. matrix, vector, scalar)', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(117, 1, '- Start event frequency evaluation:')
    worksheet.write(117, 2, str('{:.2f}'.format(start_event_frequency_evaluation(log, 'all_events', 0.01))))
    worksheet.write(117, 3, str('{:.2f}'.format(start_event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(119, 1, '- End event frequency evaluation:')
    worksheet.write(119, 2, str('{:.2f}'.format(end_event_frequency_evaluation(log, 'all_events', 0.01))))
    worksheet.write(119, 3, str('{:.2f}'.format(end_event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(121, 1, '- Event frequency evaluation:')
    worksheet.write(121, 2, str('{:.2f}'.format(event_frequency_evaluation(log, 'all_events', 0.01))))
    worksheet.write(121, 3, str('{:.2f}'.format(event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(121, 4, str('{:.2f}'.format(event_frequency_evaluation(log, 'boxplot', 0.00))))
    worksheet.write(123, 1, '- Trace frequency evaluation:')
    worksheet.write(123, 2, str('{:.2f}'.format(trace_frequency_evaluation(log, 'all_events', 0.01))))
    worksheet.write(123, 3, str('{:.2f}'.format(trace_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(123, 4, str('{:.2f}'.format(trace_frequency_evaluation(log, 'boxplot', 0.00))))
    worksheet.write(125, 1, '- Event dependency evaluation:')
    worksheet.write(125, 2, str('{:.2f}'.format(event_dependency_evaluation(log))))
    worksheet.write(127, 1, '- Trace length evaluation:')
    worksheet.write(127, 2, str('{:.2f}'.format(trace_length_evaluation(log))))
    outlier = number_of_outlying_traces(log)
    worksheet.write(129, 1, '- Number of outlying traces:')
    worksheet.write(129, 2, str(outlier))
    worksheet.write(131, 1, '- Relative number of outlying traces:')
    worksheet.write(131, 2, str('{:.2f}'.format(outlier / total_number_of_traces(log))))
    worksheet.write(133, 1, '- Event profile average euclidean distance:')
    worksheet.write(133, 2, str('{:.2f}'.format(event_profile_average_euclidean_distance(log))))
    worksheet.write(135, 1, '- Event profile average cosine similarity:')
    worksheet.write(135, 2, str('{:.2f}'.format(event_profile_average_cosine_similarity(log))))
    worksheet.write(137, 1, '- Transition profile average euclidean distance:')
    worksheet.write(137, 2, str('{:.2f}'.format(transition_profile_average_euclidean_distance(log))))
    worksheet.write(139, 1, '- Transition profile average cosine similarity:')
    worksheet.write(139, 2, str('{:.2f}'.format(transition_profile_average_cosine_similarity(log))))
    worksheet.write(141, 1, '- Event profile minimum cosine similarity:')
    worksheet.write(141, 2, str('{:.2f}'.format(event_profile_minimum_cosine_similarity(log))))
    worksheet.write(143, 1, '- Transition profile minimum cosine similarity:')
    worksheet.write(143, 2, str('{:.2f}'.format(transition_profile_minimum_cosine_similarity(log))))
    worksheet.write(145, 1, '- Average spatial proximity:')
    worksheet.write(145, 2, str('{:.2f}'.format(average_spatial_proximity(log))))
    worksheet.write(147, 1, '- Spatial proximity connectedness:')
    worksheet.write(147, 2, str('{:.2f}'.format(spatial_proximity_connectedness(log))))
    worksheet.write(149, 1, '- Spatial proximity abstraction evaluation:')
    worksheet.write(149, 2, str('{:.2f}'.format(spatial_proximity_abstraction_evaluation(log, avg=True))))
    worksheet.write(151, 1, '- Event dependency abstraction evaluation:')
    worksheet.write(151, 2, str('{:.2f}'.format(event_dependency_abstraction_evaluation(log))))
    worksheet.write(153, 1, '- Triple abstraction evaluation :')
    worksheet.write(153, 2, str('{:.2f}'.format(triple_abstraction_evaluation(log))))
    worksheet.write(155, 1, '- Event class triple abstraction evaluation:')
    worksheet.write(155, 2, str('{:.2f}'.format(event_class_triple_abstraction_evaluation(log))))

    print('Self-developed measures - derived from linear structures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(157, 1, 'Derived from non-linear features (i.e. graph)', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(159, 1, '- Number of graph communities:')
    worksheet.write(159, 2, str(number_of_graph_communities(log)))
    worksheet.write(161, 1, '- Cut-vertex outgoing degree:')
    worksheet.write(161, 2, str(maximum_cut_vertex_outgoing_degree(log)))
    worksheet.write(163, 1, '- Maximum cut-vertex independent path:')
    worksheet.write(163, 2, str(cut_vertex_independent_path(log)))
    worksheet.write(165, 1, '- Simple path minimum jaccard similarity:')
    worksheet.write(165, 2, str('{:.2f}'.format(simple_path_minimum_jaccard_similarity(log))))
    worksheet.write(167, 1, '- Syntactic node similarity:')
    worksheet.write(167, 2, str('{:.2f}'.format(syntactic_node_similarity(log))))

    print('Self-developed measures - derived from non-linear structures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------

    workbook.close()

    t2 = time()
    time_needed = timedelta(seconds=(t2 - t1))
    print()

    return 'Measure Evaluation Completed! - Time needed: ' + str(time_needed)
