# METHODS EVALUATING THE EVENT LOG'S EVENT PROFILE
from tqdm import tqdm
import numpy as np


# Average distance per variant
def event_profile_distance_all_variants(log, distance_measure):
    from encoding import event_profile_encoding_all_variants
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes, average_trace_length

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Event profile distance all variants, completed :: ")

    # Calculate distance for every variant
    encoding = event_profile_encoding_all_variants(log)
    distance_per_variant_list = []
    if distance_measure == "cosine":
        for reference_var in encoding:
            progress.update()
            dist = sum(cosine(reference_var, compared_var) for compared_var in encoding)
            distance_per_variant_list.append(dist)
    if distance_measure == "euclidean":
        for reference_var in encoding:
            progress.update()
            dist = sum(euclidean(reference_var, compared_var) for compared_var in encoding)
            distance_per_variant_list.append(dist)

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_variant_list), sum(distance_per_variant_list) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1)) / average_trace_length(log), (sum(distance_per_variant_list) - total_number_of_trace_classes(log)) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1))


# Average distance per case
def event_profile_distance_all_cases(log, distance_measure):
    from encoding import event_profile_encoding_all_traces
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces, average_trace_length

    # Set up progress bar
    no_trace = len(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Event profile distance all cases, completed :: ")

    # Calculate distance for every case
    encoding = event_profile_encoding_all_traces(log)
    distance_per_trace_list = []
    if distance_measure == "cosine":
        for reference_trace in encoding:
            progress.update()
            dist = sum(cosine(reference_trace, compared_trace) for compared_trace in encoding)
            distance_per_trace_list.append(dist)
    if distance_measure == "euclidean":
        for reference_trace in encoding:
            progress.update()
            dist = sum(euclidean(reference_trace, compared_trace) for compared_trace in encoding)
            distance_per_trace_list.append(dist)

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_trace_list), sum(distance_per_trace_list) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1)) / average_trace_length(log), (sum(distance_per_trace_list) - total_number_of_traces(log)) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1))


# Average distance per variant
def event_profile_minimum_cosine_all_variants(log):
    from encoding import event_profile_encoding_all_variants
    from fastdist.fastdist import cosine
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="event profile distance all variants, completed :: ")

    # Calculate distance for every variant
    encoding = event_profile_encoding_all_variants(log)
    min_dist = 1
    for reference_var in encoding:
        progress.update()
        for compared_var in encoding:
            dist = cosine(reference_var, compared_var)
            if min_dist > dist:
                min_dist = dist

    # Close progress bar and return
    progress.close()
    del progress
    return min_dist

