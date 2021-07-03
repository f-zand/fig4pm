# METHODS EVALUATING THE EVENT LOG'S TRACES
from tqdm import tqdm
import numpy as np


# Average distance per variant
def trace_distance_all_variants(log):
    from general_methods import variant_list
    from editdistance import distance
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Trace distance all variants, completed :: ")

    # Calculate editdistance for every variant
    variant_list = variant_list(log)
    distance_per_variant_list = []
    for reference_var in variant_list:
        progress.update()
        dist = sum(distance(reference_var, compared_var) for compared_var in variant_list)
        distance_per_variant_list.append(dist)

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_variant_list), sum(distance_per_variant_list) / (total_number_of_trace_classes(log) ** 2)


# Average distance per case
def trace_distance_all_cases(log):
    from general_methods import case_list
    from editdistance import distance
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces

    # Set up progress bar
    no_trace = len(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="trace distance all cases, completed :: ")

    # Calculate editdistance for every case
    case_list = case_list(log)
    distance_per_case_list = []
    for reference_trace in case_list:
        progress.update()
        dist = sum(distance(reference_trace, compared_trace) for compared_trace in case_list)
        distance_per_case_list.append(dist)

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_case_list), sum(distance_per_case_list) / (total_number_of_traces(log) ** 2)
