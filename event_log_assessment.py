# FINAL EVENT LOG ASSESSMENT TOOL
# STRUCTURED ALONG THE CLASSES ELABORATED THROUGHOUT THE EVALUATION STAGE

# Import relevant modules
import xlsxwriter
from time import time
from datetime import timedelta
from measures_extracted_from_literature.derived_from_linear_structures import *
from measures_extracted_from_literature.derived_from_non_linear_structures import *
from self_developed_measures.derived_from_linear_structures import *
from self_developed_measures.derived_from_non_linear_structures import *


import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import numpy as np
# from measures_extracted_from_literature.derived_from_non_linear_structures import graph_diameter, cyclicity
#
# from self_developed_measures.derived_from_linear_structures import event_class_triple_abstraction_evaluation
# Measure evaluation
def event_log_assessment(log):

    print()
    print('Log Imported!')
    print()
    t1 = time()

    # if total_number_of_trace_classes(log) == 1:  # assessment does not work for event logs that only contain one variant
    #     raise Exception('Evaluation not possible - Log only contains one variant!')

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
    worksheet.write(2, 1, 'First Level Measures:', bold)
    worksheet.write(3, 1, '-------------------------------------------------------------------', bold)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(5, 1, '- Total number of events:')
    worksheet.write(5, 2, str(total_number_of_events(log)))
    worksheet.write(7, 1, '- Total number of event classes:')
    worksheet.write(7, 2, str(total_number_of_event_classes(log)))
    worksheet.write(9, 1, '- Total number of traces:')
    worksheet.write(9, 2, str(total_number_of_traces(log)))
    worksheet.write(11, 1, '- Total number of trace classes (i.e. variants):')
   # worksheet.write(11, 2, str(total_number_of_trace_classes(log)))
    worksheet.write(13, 1, '- Minimum trace length:')
    worksheet.write(13, 2, str(minimum_trace_length(log)))
    worksheet.write(15, 1, '- Maximum trace length:')
    worksheet.write(15, 2, str(maximum_trace_length(log)))
    worksheet.write(17, 1, '- Absolute number of distinct start events:')
    worksheet.write(17, 2, str(number_of_distinct_start_events(log)))
    worksheet.write(19, 1, '- Absolute number of distinct end events:')
    worksheet.write(19, 2, str(number_of_distinct_end_events(log)))
    worksheet.write(21, 1, '- Absolute number of traces with a self-loop:')
    worksheet.write(21, 2, str(absolute_number_of_traces_with_self_loop(log)))
    worksheet.write(23, 1, '- Absolute number of traces with a repetition:')
    worksheet.write(23, 2, str(absoulute_number_of_traces_with_repetition(log)))
    worksheet.write(25, 1, '- Absolute trace coverage:')
    worksheet.write(25, 2, str(absolute_trace_coverage(log)))
    worksheet.write(26, 1, '  Explanatory information: number of traces required to cover 80% of the log\'s traces')
    worksheet.write(28, 1, '- Number of nodes:')
    worksheet.write(28, 2, str(number_of_nodes(log)))
    worksheet.write(30, 1, '- Number of arcs:')
    worksheet.write(30, 2, str(number_of_arcs(log)))

    print('First Level Measures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(32, 1, '-------------------------------------------------------------------', bold)
    worksheet.write(33, 1, 'Second Level Measures:', bold)
    worksheet.write(34, 1, '-------------------------------------------------------------------', bold)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(36, 1, 'Outlier Detection Measures', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(38, 1, '- Coefficient of network connectivity:')
    worksheet.write(38, 2, str('{:.2f}'.format(coefficient_of_network_connectivity(log))))
    worksheet.write(39, 1, '  Explanatory information: number of edges per node')
    worksheet.write(40, 1, '  Significance: High')
    worksheet.write(42, 1, '- Average node degree:')
    worksheet.write(42, 2, str('{:.2f}'.format(average_node_degree(log))))
    worksheet.write(43, 1, '  Significance: High')
    worksheet.write(45, 1, '- Maximum node degree:')
    worksheet.write(45, 2, str('{:.2f}'.format(maximum_node_degree(log))))
    worksheet.write(46, 1, '  Significance: Medium')
    worksheet.write(48, 1, '- Start event frequency evaluation:')
    worksheet.write(48, 2, str('{:.2f}'.format(start_event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(49, 1, '  Explanatory information: percentage of start events that occur less than 5% when compared to the maximum frequency of any start event in the log')
    worksheet.write(50, 1, '  Significance: Medium')
    worksheet.write(52, 1, '- End event frequency evaluation:')
    worksheet.write(52, 2, str('{:.2f}'.format(end_event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(53, 1, '  Explanatory information: percentage of end events that occur less than 5% when compared to the maximum frequency of any end event in the log')
    worksheet.write(54, 1, '  Significance: Medium')
    worksheet.write(56, 1, '- Event frequency evaluation:')
    worksheet.write(56, 2, str('{:.2f}'.format(event_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(57, 1, '  Explanatory information: percentage of events that occur less than 5% when compared to the maximum frequency of any event in the log')
    worksheet.write(58, 1, '  Significance: Medium')
    worksheet.write(60, 1, '- Trace frequency evaluation:')
    worksheet.write(60, 2, str('{:.2f}'.format(trace_frequency_evaluation(log, 'highest_occurrence', 0.05))))
    worksheet.write(61, 1, '  Explanatory information: percentage of variants occurring less than 5% when compared to the maximum frequency of any variant in the log')
    worksheet.write(62, 1, '  Significance: Medium')
    worksheet.write(64, 1, '- Event dependency evaluation:')
    worksheet.write(64, 2, str('{:.2f}'.format(event_dependency_evaluation(log))))
    worksheet.write(65, 1, '  Explanatory information: Transitions between events are perceived as outlying behavior if they exhibit a low dependence')
    worksheet.write(66, 1, '  Significance: Medium')
    worksheet.write(68, 1, '- Trace length evaluation:')
    worksheet.write(68, 2, str('{:.2f}'.format(trace_length_evaluation(log))))
    worksheet.write(69, 1, '  Explanatory information: percentage of cases highly deviating from the median trace length')
    worksheet.write(70, 1, '  Significance: High')
#    outlier = number_of_outlying_traces(log)
    worksheet.write(72, 1, '- Number of outlying traces:')
 #   worksheet.write(72, 2, str(outlier))
    worksheet.write(73, 1, '  Explanatory information: algorithm used is IFOREST with a contamination factor of 5%')
    worksheet.write(74, 1, '  Significance: Medium')
    worksheet.write(76, 1, '- Relative number of outlying traces:')
#    worksheet.write(76, 2, str('{:.2f}'.format(outlier / total_number_of_traces(log))))
    worksheet.write(77, 1, '  Explanatory information: algorithm used is IFOREST with a contamination factor of 5%')
    worksheet.write(78, 1, '  Significance: Medium')

    print('Second Level Outlier Removal Measures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(80, 1, 'Trace Clustering Measures', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(82, 1, '- Simple trace diversity:')
    worksheet.write(82, 2, str('{:.2f}'.format(simple_trace_diversity(log))))
    worksheet.write(83, 1, '  Explanatory information: value converges to 0 in case the logs variants only arise from different ordering of events - value converges to 1 in case the logs variants arise from a fraction of events observed in the log')
    worksheet.write(84, 1, '  Significance: High')
    worksheet.write(86, 1, '- Number of cut-vertices:')
    worksheet.write(86, 2, str('{:.2f}'.format(number_of_cut_vertices(log))))
    worksheet.write(87, 1, '  Explanatory information: number of vertices whose removal separates the directly-follows graph into several components')
    worksheet.write(88, 1, '  Significance: Medium')
    worksheet.write(90, 1, '- Separability ratio:')
    worksheet.write(90, 2, str('{:.2f}'.format(separability_ratio(log))))
    worksheet.write(91, 1, '  Explanatory information: number of cut-vertices related to total number of vertices')
    worksheet.write(92, 1, '  Significance: Low')
    worksheet.write(94, 1, '- Event profile minimum cosine similarity:')
    worksheet.write(94, 2, str('{:.2f}'.format(event_profile_minimum_cosine_similarity(log))))
    worksheet.write(95, 1, '  Explanatory information: tends towards 0 in case the log contains variants with a highly deviating composition of events')
    worksheet.write(96, 1, '  Significance: Medium')
    worksheet.write(98, 1, '- Average spatial proximity:')
    worksheet.write(98, 2, str('{:.2f}'.format(average_spatial_proximity(log))))
    worksheet.write(99, 1, '  Explanatory information: indicates how densely connected all events in the log are - tends towards 0 for more unconnected events')
    worksheet.write(100, 1, '  Significance: High')
    worksheet.write(102, 1, '- Spatial proximity connectedness:')
    worksheet.write(102, 2, str('{:.2f}'.format(spatial_proximity_connectedness(log))))
    worksheet.write(103, 1, '  Explanatory information: indicates how densely connected all events in the log are - tends towards 0 for more unconnected events - better suited than \'average spatial proximity measure\'')
    worksheet.write(104, 1, '  Significance: High')
    worksheet.write(106, 1, '- Number of graph communities:')
    worksheet.write(106, 2, str(number_of_graph_communities(log)))
    worksheet.write(107, 1, '  Significance: Medium')
    worksheet.write(109, 1, '- Maximum cut-vertex outgoing degree:')
    worksheet.write(109, 2, str(maximum_cut_vertex_outgoing_degree(log)))
    worksheet.write(110, 1, '  Significance: Medium')
    worksheet.write(112, 1, '- Cut-vertex independent path:')
    worksheet.write(112, 2, str(cut_vertex_independent_path(log)))
    worksheet.write(113, 1, '  Explanatory information: highest number of node-independent paths between any two cut-vertices in the directly-follows graph')
    worksheet.write(114, 1, '  Significance: Medium')
    worksheet.write(116, 1, '- Simple path minimum jaccard similarity:')
    worksheet.write(116, 2, str('{:.2f}'.format(simple_path_minimum_jaccard_similarity(log))))
    worksheet.write(117, 1, '  Explanatory information: tends towards 0 in case the directly-follows graph contains simple paths with a highly deviating composition of nodes')
    worksheet.write(118, 1, '  Significance: High')

    print('Second Level Trace Clustering Measures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(120, 1, 'Event Abstraction Measures', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(122, 1, '- Sequentiality ratio:')
    worksheet.write(122, 2, str('{:.2f}'.format(sequentiality_ratio(log))))
    worksheet.write(123, 1, '  Explanatory information: relative number of non-connector nodes contained in the directly-follows graph')
    worksheet.write(124, 1, '  Significance: Medium')
    worksheet.write(126, 1, '- Spatial proximity abstraction evaluation:')
    worksheet.write(126, 2, str('{:.2f}'.format(spatial_proximity_abstraction_evaluation(log, avg=True))))
    worksheet.write(127, 1, '  Explanatory information: percentage of highly connected event pairs in terms of their spatial proximity')
    worksheet.write(128, 1, '  Significance: Low')
    worksheet.write(130, 1, '- Event dependency abstraction evaluation:')
    worksheet.write(130, 2, str('{:.2f}'.format(event_dependency_abstraction_evaluation(log))))
    worksheet.write(131, 1, '  Explanatory information: percentage of highly connected event pairs in terms of their event dependency')
    worksheet.write(132, 1, '  Significance: High')
    worksheet.write(134, 1, '- Triple abstraction evaluation :')
    worksheet.write(134, 2, str('{:.2f}'.format(triple_abstraction_evaluation(log))))
    worksheet.write(135, 1, '  Explanatory information: percentage of triples containing the same intermediate event')
    worksheet.write(136, 1, '  Significance: Low')
    worksheet.write(138, 1, '- Event class triple abstraction evaluation:')
    worksheet.write(138, 2, str('{:.2f}'.format(event_class_triple_abstraction_evaluation(log))))
    worksheet.write(139, 1, '  Explanatory information: percentage of event classes occurring as the same intermediate event in all event triples')
    worksheet.write(140, 1, '  Significance: Low')
    worksheet.write(142, 1, '- Syntactic node similarity:')
#    worksheet.write(142, 2, str('{:.2f}'.format(syntactic_node_similarity(log))))
    worksheet.write(143, 1, '  Explanatory information: Percentage of event classes exhibiting a syntactic node similarity higher than 50%')
    worksheet.write(144, 1, '  Significance: High')

    print('Second Level Event Abstraction Measures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(146, 1, 'Supporting Measures', underline)
    # ----------------------------------------------------------------------------------------------------------------
    worksheet.write(148, 1, '- Average trace length:')
    worksheet.write(148, 2, str('{:.2f}'.format(average_trace_length(log))))
    worksheet.write(150, 1, '- Average trace size:')
    worksheet.write(150, 2, str('{:.2f}'.format(average_trace_size(log))))
    worksheet.write(152, 1, '- Relative number of distinct start events:')
    worksheet.write(152, 2, str('{:.2f}'.format(relative_number_of_distinct_start_events(log))))
    worksheet.write(154, 1, '- Relative number of distinct end events:')
    worksheet.write(154, 2, str('{:.2f}'.format(relative_number_of_distinct_end_events(log))))
    worksheet.write(156, 1, '- Relative number of traces with a self-loop:')
    worksheet.write(156, 2, str('{:.2f}'.format(relative_number_of_traces_with_self_loop(log))))
    worksheet.write(158, 1, '- Average number of self-loops per trace:')
    worksheet.write(158, 2, str('{:.2f}'.format(average_number_of_self_loops_per_trace(log))))
    worksheet.write(160, 1, '- Maximum number of self-loops per trace:')
    worksheet.write(160, 2, str('{:.2f}'.format(maximum_number_of_self_loops_per_trace(log))))
    worksheet.write(162, 1, '- Average size of self-loops per trace:')
    worksheet.write(162, 2, str('{:.2f}'.format(average_size_of_self_loops_per_trace(log))))
    worksheet.write(164, 1, '- Maximum size of self-loops per trace:')
    worksheet.write(164, 2, str('{:.2f}'.format(maximum_size_of_self_loops_per_trace(log))))
    worksheet.write(166, 1, '- Relative number of traces with a repetition:')
    worksheet.write(166, 2, str('{:.2f}'.format(relative_number_of_traces_with_repetition(log))))
    worksheet.write(168, 1, '- Number of distinct traces per 100 traces:')
    worksheet.write(168, 2, str('{:.2f}'.format(number_of_distinct_traces_per_hundred_traces(log))))
    worksheet.write(170, 1, '- Relative trace coverage:')
    worksheet.write(170, 2, str('{:.2f}'.format(relative_trace_coverage(log))))
    worksheet.write(171, 1, '  Explanatory information: relative number of traces required to cover 80% of the log\'s traces')
    worksheet.write(173, 1, '- Event density:')
    worksheet.write(173, 2, str('{:.2f}'.format(event_density(log))))
    worksheet.write(174, 1, '  Explanatory information: value converges to 1 in case the size of the log (by means of the average trace length) arises from sustained sequences - value converges to 0 in case the size of the log (by means of the average trace length) arises from recurring behavior caused by loops')
    worksheet.write(176, 1, '- Trace heterogeneity rate (log normalized):')
    worksheet.write(176, 2, str('{:.2f}'.format(traces_heterogeneity_rate(log))))
    worksheet.write(177, 1, '  Explanatory information: relates the number of trace classes to the number of all traces in the event log')
    worksheet.write(179, 1, '- Trace similarity rate:')
#    worksheet.write(179, 2, str('{:.2f}'.format(trace_similarity_rate(log))))
    worksheet.write(180, 1, '  Explanatory information: similarity between all traces in terms of levenshtein edit distance')
    worksheet.write(182, 1, '- Complexity factor:')
#    worksheet.write(182, 2, str('{:.2f}'.format(complexity_factor(log))))
    worksheet.write(183, 1, '  Explanatory information: investigates the understandability of the future model mined from the given event log. The higher the complexity factor, the more complex the discovered model will be')
    worksheet.write(184, 1, '  Complexity Factor Ranking:')
    worksheet.write(185, 1, '  0 - 30: Simple')
    worksheet.write(186, 1, '  30 - 53: Somewhat Complex')
    worksheet.write(187, 1, '  53 - 75: Complex')
    worksheet.write(188, 1, '  75 - 95: Very Complex')
    worksheet.write(189, 1, '  > 95: Highly Complex')
#    advanced_tr_div = advanced_trace_diversity(log)
    worksheet.write(191, 1, '- Advanced trace diversity (full steps):')
#    worksheet.write(191, 2, str('{:.2f}'.format(advanced_tr_div * average_trace_length(log))))
    worksheet.write(192, 1, '  Explanatory information: mean levenshtein distance between any two variants in the event log')
    worksheet.write(194, 1, '- Advanced trace diversity (relative):')
#    worksheet.write(194, 2, str('{:.2f}'.format(advanced_tr_div)))
    worksheet.write(195, 1, '  Explanatory information: mean extent of modifications necessary to transform an arbitrary trace to any other trace within the log')
    worksheet.write(197, 1, '- Trace entropy:')
    worksheet.write(197, 2, str('{:.2f}'.format(trace_entropy(log))))
    worksheet.write(198, 1, '  Explanatory information: entropy based on variant frequency - highly structured processes should generate more homogeneous (low entropy) traces and more flexible processes should generate more varied (high entropy) traces')
    worksheet.write(200, 1, '- Prefix entropy (flattened):')
    worksheet.write(200, 2, str('{:.2f}'.format(prefix_entropy(log))))
    worksheet.write(201, 1, '  Explanatory information: entropy based on frequencies of all prefixes in the log')
    worksheet.write(203, 1, '- All-block entropy (flattened):')
    worksheet.write(203, 2, str('{:.2f}'.format(all_block_entropy(log))))
    worksheet.write(204, 1, '  Explanatory information: entropy based on frequencies of all blocks in the log')
    worksheet.write(206, 1, '- Graph Density:')
    worksheet.write(206, 2, str('{:.2f}'.format(density(log))))
    worksheet.write(207, 1, '  Explanatory information: ratio between existing arcs and possible arcs')
    worksheet.write(209, 1, '- Structure:')
    worksheet.write(209, 2, str('{:.2f}'.format(structure(log))))
    worksheet.write(210, 1, '  Explanatory information: values tending towards 0 depict low structure of the log')
    worksheet.write(212, 1, '- Cyclomatic number of the graph:')
    worksheet.write(212, 2, str('{:.2f}'.format(cyclomatic_number(log))))
    worksheet.write(213, 1, '  Explanatory information: maximum number of linearly independent cycles of the graph')
    worksheet.write(215, 1, '- Graph diameter:')
    worksheet.write(215, 2, str('{:.2f}'.format(graph_diameter(log))))
    worksheet.write(216, 1, '  Explanatory information: length of the longest path from a start to an end node in the process model')
    worksheet.write(218, 1, '- Cyclicity:')
    worksheet.write(218, 2, str('{:.2f}'.format(cyclicity(log))))
    worksheet.write(219, 1, '  Explanatory information: relative amount of nodes that are part of a cycle')
    worksheet.write(221, 1, '- Affinity:')
    worksheet.write(221, 2, str('{:.2f}'.format(affinity(log))))
    worksheet.write(222, 1, '  Explanatory information: quantifies the average similarity between traces regarding their directly-follows relations')
    worksheet.write(224, 1, '- Simple Path Complexity:')
    worksheet.write(224, 2, str(simple_path_complexity(log)))
    worksheet.write(225, 1, '  Explanatory information: number of paths without cycles from source to sink')
    worksheet.write(227, 1, '- Event profile average euclidean distance:')
    worksheet.write(227, 2, str('{:.2f}'.format(event_profile_average_euclidean_distance(log))))
    worksheet.write(229, 1, '- Event profile average cosine similarity:')
    worksheet.write(229, 2, str('{:.2f}'.format(event_profile_average_cosine_similarity(log))))
    worksheet.write(231, 1, '- Transition profile average euclidean distance:')
    worksheet.write(231, 2, str('{:.2f}'.format(transition_profile_average_euclidean_distance(log))))
    worksheet.write(233, 1, '- Transition profile average cosine similarity:')
    worksheet.write(233, 2, str('{:.2f}'.format(transition_profile_average_cosine_similarity(log))))
    worksheet.write(235, 1, '- Transition profile minimum cosine similarity:')
    worksheet.write(235, 2, str('{:.2f}'.format(transition_profile_minimum_cosine_similarity(log))))

    print('Second Level Supporting Measures - Done! ')

    # ----------------------------------------------------------------------------------------------------------------

    workbook.close()

    t2 = time()
    time_needed = timedelta(seconds=(t2 - t1))
    print()

    return 'Measure Evaluation Completed! - Time needed: ' + str(time_needed)
#
log_csv = pd.read_csv('BPI_2019.csv', sep=',')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('event time:timestamp')
log_csv.rename(columns={"case concept:name": "case:concept:name"}, inplace=True)

# find variants to perform sampling (optional)
log_csv2 = log_csv.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index(name = 'variants')

#define total sample size desired (optional)
N = round(0.01 * log_csv['case:concept:name'].nunique())
log_csv3 = log_csv.copy()
#perform stratified random sampling (optional)
log_csv2 = log_csv2.groupby('variants', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(log_csv2))))).sample(frac=1).reset_index(drop=True)
# count events per case (mandatory because of the exception in the code)
log_csv3 = log_csv3.groupby('case:concept:name')['concept:name'].count().reset_index(name="count")
# filter out cases with only one event
log_csv3= log_csv3[log_csv3['count'] > 1]
#get final dataset
log_csv4 = log_csv.merge(log_csv2, how = "inner", on = "case:concept:name").drop(columns=['variants'])
log_csv4 = log_csv4.merge(log_csv3, how = "inner", on = "case:concept:name")

# convert dataframe to log
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
event_log = log_converter.apply(log_csv4, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
# run the program
event_log_assessment(event_log)

# # when file is .xes
# from pm4py.objects.log.importer.xes import importer as xes_importer
# log = xes_importer.apply('ExampleLog.xes')
# event_log_assessment(log)