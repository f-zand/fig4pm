# METHODS IMPLEMENTING SELF-DEVELOPED MEASURES
# DERIVED FROM LINEAR STRUCTURES OF THE EVENT LOG


# Outlier evaluation of start event frequencies
def start_event_frequency_evaluation(log, lower_bound, threshold):
    from pm4py.statistics.start_activities.log import get
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_events

    start_event_frequency = list(get.get_start_activities(log).values())
    if lower_bound == 'all_events':
        bound = total_number_of_events(log) * threshold
    if lower_bound == 'highest_occurrence':
        bound = max(start_event_frequency) * threshold
    counter = 0
    for i in start_event_frequency:
        if i < bound:
            counter += 1
    return counter / len(start_event_frequency)


# Outlier evaluation of end event frequencies
def end_event_frequency_evaluation(log, lower_bound, threshold):
    from pm4py.statistics.end_activities.log import get
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_events

    end_event_frequency = list(get.get_end_activities(log).values())
    if lower_bound == 'all_events':
        bound = total_number_of_events(log) * threshold
    if lower_bound == 'highest_occurrence':
        bound = max(end_event_frequency) * threshold
    counter = 0
    for i in end_event_frequency:
        if i < bound:
            counter += 1
    return counter / len(end_event_frequency)


# Outlier evaluation of event frequencies
def event_frequency_evaluation(log, lower_bound, threshold):
    from pm4py.statistics.attributes.log.get import get_attribute_values
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_events
    from numpy import quantile

    event_frequency = list(get_attribute_values(log, 'concept:name').values())
    if lower_bound == 'all_events':
        bound = total_number_of_events(log) * threshold
    if lower_bound == 'highest_occurrence':
        bound = max(event_frequency) * threshold
    if lower_bound == 'boxplot':
        quantile_25, quantile_50, quantile_75 = quantile(event_frequency, [0.25, 0.5, 0.75])
        iqr = quantile_75 - quantile_25
        bound = quantile_25 - (1.5 * iqr)
    counter = 0
    for i in event_frequency:
        if i < bound:
            counter += 1
    return counter / len(event_frequency)


# Outlier evaluation of trace frequencies
def trace_frequency_evaluation(log, lower_bound, threshold):
    from general_methods import variant_count_list
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces
    from numpy import quantile

    trace_frequency = variant_count_list(log)
    if lower_bound == 'all_events':
        bound = total_number_of_traces(log) * threshold
    if lower_bound == 'highest_occurrence':
        bound = max(trace_frequency) * threshold
    if lower_bound == 'boxplot':
        quantile_25, quantile_50, quantile_75 = quantile(trace_frequency, [0.25, 0.5, 0.75])
        iqr = quantile_75 - quantile_25
        bound = quantile_25 - (1.5 * iqr)
    counter = 0
    for i in trace_frequency:
        if i < bound:
            counter += 1
    return counter / len(trace_frequency)


# Outlier evaluation of event dependency
def event_dependency_evaluation(log, threshold=0.05):
    from event_dependency import event_dependency_matrix
    import numpy as np

    event_dependency_list = event_dependency_matrix(log).flatten()
    nonzero = len(event_dependency_list[np.nonzero(event_dependency_list)])
    counter = 0
    for elem in event_dependency_list:
        if 0 < elem < threshold:
            counter += 1
    return counter / nonzero


# Outlier evaluation of trace length
def trace_length_evaluation(log):
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces
    from outlier_detection import box_whisker_plot_evaluation

    trace_length_list = []
    for case in log:
        trace_length_list.append(len(case))
    outliers = box_whisker_plot_evaluation(trace_length_list)
    return (outliers[0] + outliers[1]) / total_number_of_traces(log)


# Absolute number of outlying traces detected via unsupervised outlier detection algorithm
def number_of_outlying_traces(log):
    from outlier_detection import outlier_detection_feature_based_unsupervised
    return outlier_detection_feature_based_unsupervised(log, 'IForest', 0.05, 0.9)[0]


# Relative number of outlying traces detected via unsupervised outlier detection algorithm
def relative_number_of_outlying_traces(log):
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces
    from outlier_detection import outlier_detection_feature_based_unsupervised
    return outlier_detection_feature_based_unsupervised(log, 'IForest', 0.05, 0.9)[0] / total_number_of_traces(log)


# Event profile average euclidean distance
def event_profile_average_euclidean_distance(log):
    from profile_distances.event_profile import event_profile_distance_all_variants
    return event_profile_distance_all_variants(log, 'euclidean')[1]


# Event profile average cosine distance
def event_profile_average_cosine_similarity(log):
    from profile_distances.event_profile import event_profile_distance_all_variants
    return event_profile_distance_all_variants(log, 'cosine')[2]


# Transition profile average euclidean distance
def transition_profile_average_euclidean_distance(log):
    from profile_distances.k_gram_profile import k_gram_distance_all_variants
    return k_gram_distance_all_variants(log, 2, 'euclidean')[1]


# Transition profile average cosine distance
def transition_profile_average_cosine_similarity(log):
    from profile_distances.k_gram_profile import k_gram_distance_all_variants
    return k_gram_distance_all_variants(log, 2, 'cosine')[2]


# Event profile maximum cosine distance
def event_profile_minimum_cosine_similarity(log):
    from profile_distances.event_profile import event_profile_minimum_cosine_all_variants
    return event_profile_minimum_cosine_all_variants(log)


# Transition profile maximum cosine distance
def transition_profile_minimum_cosine_similarity(log):
    from profile_distances.k_gram_profile import k_gram_minimum_cosine_all_variants
    return k_gram_minimum_cosine_all_variants(log, 2)


# Average spatial proximity
def average_spatial_proximity(log):
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_event_classes
    from spatial_proximity import spatial_proximity_matrix

    spatial_proximity_list = spatial_proximity_matrix(log).flatten()
    return sum(spatial_proximity_list) / (total_number_of_event_classes(log) * (total_number_of_event_classes(log) - 1))


# Spatial proximity connectedness
def spatial_proximity_connectedness(log):
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_event_classes
    from spatial_proximity import spatial_proximity_matrix

    spatial_proximity_list = spatial_proximity_matrix(log).flatten()
    counter = 0
    for elem in spatial_proximity_list:
        if elem == 0:
            counter += 1
    return 1 - ((counter - total_number_of_event_classes(log)) / (total_number_of_event_classes(log) * (total_number_of_event_classes(log) - 1)))


# Spatial proximity abstraction evaluation
def spatial_proximity_abstraction_evaluation(log, avg=True, threshold=0.9):
    from measures_extracted_from_literature.derived_from_linear_structures import average_trace_length
    from spatial_proximity import spatial_proximity_matrix

    if avg == True:
        avg_trace_len = average_trace_length(log)
        thres = 1 - (1 / avg_trace_len)
    else:
        thres = threshold
    spatial_proximity_list = spatial_proximity_matrix(log).flatten()
    nonzero = len(spatial_proximity_list[np.nonzero(spatial_proximity_list)])
    counter = 0
    for elem in spatial_proximity_list:
        if elem >= thres:
            counter += 1
    return counter / nonzero


# Event dependency abstraction evaluation
def event_dependency_abstraction_evaluation(log):
    from event_dependency import event_dependency_matrix

    event_dependency_list = event_dependency_matrix(log).flatten()
    nonzero = len(event_dependency_list[np.nonzero(event_dependency_list)])
    counter = 0
    for elem in event_dependency_list:
        if elem == 0.5:
            counter += 1
    return counter / nonzero


# Triple abstraction evaluation
def triple_abstraction_evaluation(log):
    from pm4py.objects.dfg.retrieval.log import freq_triples

    triples = list(freq_triples(log).keys())
    target_triples = set()
    for i in triples:
        for j in triples:
            if i is not j:
                if i[0] == j[0] and i[-1] == j[-1]:
                    target_triples.add(i)
                    target_triples.add(j)
    return len(target_triples) / len(triples)


# Event class triple abstraction evaluation
def event_class_triple_abstraction_evaluation(log):
    from pm4py.objects.dfg.retrieval.log import freq_triples
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_event_classes

    triples = list(freq_triples(log).keys())
    events = set()
    for i in triples:
        for j in triples:
            if i is not j:
                if i[0] == j[0] and i[-1] == j[-1]:
                    events.add(i[1])
                    events.add(j[1])
    return len(events) / total_number_of_event_classes(log)
