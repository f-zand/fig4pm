# METHODS IMPLEMENTING MEASURES EXTRACTED FROM THE LITERATURE
# DERIVED FROM LINEAR STRUCTURES OF THE EVENT LOG

# Import relevant modules
from general_methods import event_names, self_loop_per_trace_overview, repetition_per_trace_overview


# Returns the number of events in the event log (ne)
def total_number_of_events(log):
    return sum(len(case) for case in log)


# Returns the number of event classes in the event log (nec)
def total_number_of_event_classes(log):
    return len(event_names(log))


# Returns the number of traces in the event log (nt)
def total_number_of_traces(log):
    return len(log)


# Returns the number of trace classes (i.e. variants) in the event log (ntc)
def total_number_of_trace_classes(log):
    from pm4py.statistics.traces.log import case_statistics
    return len(case_statistics.get_variant_statistics(log))


# Returns the average trace length (atl)
# (i.e. total number of events in the log divided by total number of cases (traces) in the log)
def average_trace_length(log):
    return total_number_of_events(log) / total_number_of_traces(log)


# Returns length of shortest trace (mitl)
def minimum_trace_length(log):
    trace_length_list = []
    for case in log:
        trace_length_list.append(len(case))
    return min(trace_length_list)


# Returns length of longest trace (matl)
def maximum_trace_length(log):
    trace_length_list = []
    for case in log:
        trace_length_list.append(len(case))
    return max(trace_length_list)


# Average trace size (i.e. number of event classes per case (trace)) (ats)
def average_trace_size(log):
    trace_size = 0
    for case in log:
        activity_list = set()
        for event in case:
            activity_list.add(event["concept:name"])
        trace_size += len(activity_list)
    return trace_size / len(log)


# Absolute number of start events (nsec)
def number_of_distinct_start_events(log):
    from pm4py.statistics.start_activities.log import get
    return len(get.get_start_activities(log))


# Absolute number of end events (ntec)
def number_of_distinct_end_events(log):
    from pm4py.statistics.end_activities.log import get
    return len(get.get_end_activities(log))


# Absolute number of traces including a self-loop of at least length 1 (ntsl)
def absolute_number_of_traces_with_self_loop(log):
    self_loop_overview = self_loop_per_trace_overview(log)
    traces_with_self_loop = 0
    for i in self_loop_overview:
        if len(i) > 0:
            traces_with_self_loop += 1
    return traces_with_self_loop


# Absolute number of traces including a repetition (excluding loops) (ntr)
def absoulute_number_of_traces_with_repetition(log):
    repetition_overview = repetition_per_trace_overview(log)
    traces_with_repetition = 0
    for i in repetition_overview:
        if len(i) > 0:
            traces_with_repetition += 1
    return traces_with_repetition


# Relative number of start activities (rnsec)
def relative_number_of_distinct_start_events(log):
    return number_of_distinct_start_events(log) / total_number_of_event_classes(log)


# Relative number of end activities (rntec)
def relative_number_of_distinct_end_events(log):
    return number_of_distinct_end_events(log) / total_number_of_event_classes(log)


# Relative number of traces including a self-loop of at least length 1 (rntsl)
def relative_number_of_traces_with_self_loop(log):
    return absolute_number_of_traces_with_self_loop(log) / total_number_of_traces(log)


# Relative number of traces including a repetition (excluding loops) (rntr)
def relative_number_of_traces_with_repetition(log):
    return absoulute_number_of_traces_with_repetition(log) / total_number_of_traces(log)


# Average number of self-loops per trace (anslt)
def average_number_of_self_loops_per_trace(log):
    loop_overview = self_loop_per_trace_overview(log)
    loops_in_traces = 0
    for i in loop_overview:
        if len(i) > 0:
            loops_in_traces += len(i)
    return loops_in_traces / total_number_of_traces(log)


# Maximum number of self-loops per trace (manslt)
def maximum_number_of_self_loops_per_trace(log):
    loop_overview = self_loop_per_trace_overview(log)
    maximum_number_self_loops = 0
    for i in loop_overview:
        if len(i) > maximum_number_self_loops:
            maximum_number_self_loops = len(i)
    return maximum_number_self_loops


# Average size of self-loops per trace (only accounting for traces that contain a self-loop) (asslt)
def average_size_of_self_loops_per_trace(log):
    loop_overview = self_loop_per_trace_overview(log)
    self_loop_traces = 0
    loop_size_in_traces = 0
    for i in loop_overview:
        if len(i) > 0:
            self_loop_traces += 1
            loop_size_in_traces += sum(i)
    try:
        result = loop_size_in_traces / self_loop_traces
    except ZeroDivisionError:
        result = 0
    return result


# Maximum size of self-loops for any trace (masslt)
def maximum_size_of_self_loops_per_trace(log):
    loop_overview = self_loop_per_trace_overview(log)
    loop_size_in_traces = 0
    for i in loop_overview:
        if len(i) > 0:
            if max(i) > loop_size_in_traces:
                loop_size_in_traces = max(i)
    return loop_size_in_traces


# Absolute number of distinct traces per 100 traces (tcpht)
def number_of_distinct_traces_per_hundred_traces(log):
    return total_number_of_trace_classes(log) / total_number_of_traces(log) * 100


# Absolute trace coverage: 80 percent level (tco)
def absolute_trace_coverage(log):
    from general_methods import variant_count_list
    variant_count_list = variant_count_list(log)
    coverage_threshold = len(log) * 0.8
    trace_counter = 0
    distinct_traces = 0
    for i in variant_count_list:
        distinct_traces += 1
        trace_counter += i
        if trace_counter >= coverage_threshold:
            break
    return distinct_traces


# Absolute trace coverage: 80 percent level (rtco)
def relative_trace_coverage(log):
    from general_methods import variant_count_list
    variant_count_list = variant_count_list(log)
    coverage_threshold = len(log) * 0.8
    trace_counter = 0
    distinct_traces = 0
    for i in variant_count_list:
        distinct_traces += 1
        trace_counter += i
        if trace_counter >= coverage_threshold:
            break
    return distinct_traces / len(variant_count_list)


# Event density (i.e. average trace size / average trace length) (edn)
def event_density(log):
    return average_trace_size(log) / average_trace_length(log)


# Traces heterogeneity rate (i.e. ln(variant_count) / ln(case_count)) (thr)
def traces_heterogeneity_rate(log):
    from math import log as natural_log
    return natural_log(total_number_of_trace_classes(log)) / natural_log(total_number_of_traces(log))


# Trace similarity rate (tsr)
def trace_similarity_rate(log):
    from editdistance import distance
    from general_methods import variant_list
    variant_list = variant_list(log)
    dist = 0
    for reference_trace in variant_list:
        for compared_trace in variant_list:
            if reference_trace is not compared_trace:
                dist += (((max(len(reference_trace), len(compared_trace))) -
                          distance(reference_trace, compared_trace)) /
                         (max(len(reference_trace), len(compared_trace))))
    return (1 / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1))) * dist


# Complexity factor (cf)
def complexity_factor(log):
    from math import log as natural_log
    ntc, tsr, edn, ats = total_number_of_trace_classes(log), trace_similarity_rate(log), event_density(log), average_trace_size(log)
    return (natural_log(ntc) ** ((1 - tsr) + edn)) * ats


# Simple trace diversity (i.e. 1 - (average trace size / total number of activities)) (std)
def simple_trace_diversity(log):
    return 1 - (average_trace_size(log) / total_number_of_event_classes(log))


# Advanced trace diversity (i.e. weighted levenshtein distance between all traces) (atd)
def advanced_trace_diversity(log):
    from editdistance import distance
    from general_methods import case_list
    case_list = case_list(log)
    dist = sum(distance(reference_trace, compared_trace) for reference_trace in case_list
               for compared_trace in case_list)
    return (1 / (total_number_of_traces(log) * (total_number_of_traces(log) - 1) * average_trace_length(log))) * dist


# Trace entropy (tentr)
def trace_entropy(log):
    from general_methods import variant_count_list
    from scipy.stats import entropy
    # calculate probabilities of the variants
    variant_count_list = variant_count_list(log)
    n = total_number_of_traces(log)
    probabilities = [(value / n) for value in variant_count_list]
    return entropy(probabilities, base=2)


# Prefix entropy (flattened) (prentr)
def prefix_entropy(log):
    from general_methods import variant_list
    from scipy.stats import entropy
    # create dictionary including all prefixes and the number of times they occur in the log
    prefix_dict = {}
    variants_list = variant_list(log)
    for variant in variants_list:
        for i in range(1, len(variant) + 1, 1):
            prefix = ", ".join(variant[0: i])
            if prefix in prefix_dict:
                prefix_dict[prefix] += 1
            else:
                prefix_dict[prefix] = 1
    # calculate probabilities of the prefixes
    prefix_count_list = list(prefix_dict.values())
    prefix_count = sum(prefix_count_list)
    probabilities = [(value / prefix_count) for value in prefix_count_list]
    return entropy(probabilities, base=2)


# All-block entropy (flattened) (abentr)
def all_block_entropy(log):
    from general_methods import variant_list
    from scipy.stats import entropy
    # create dictionary including all blocks and the number of times they occur in the log
    all_block_dict = {}
    variants_list = variant_list(log)
    for variant in variants_list:
        for i in range(0, len(variant), 1):
            for k in range(i + 1, len(variant) + 1, 1):
                block = ", ".join(variant[i: k])
                if block in all_block_dict:
                    all_block_dict[block] += 1
                else:
                    all_block_dict[block] = 1
    # calculate probabilities of the blocks
    all_block_count_list = list(all_block_dict.values())
    all_block_count = sum(all_block_count_list)
    probabilities = [(value / all_block_count) for value in all_block_count_list]
    return entropy(probabilities, base=2)
