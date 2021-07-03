# METHODS EVALUATING THE EVENT LOG'S TRANSITION PROFILE
from tqdm import tqdm
import numpy as np


# METHODS FOR GENERAL TRANSITION PROFILE

# Average distance per variant
def transition_profile_distance_all_variants(log, profile, distance_measure):
    from encoding import transition_profile_encoding_all_variants_matrix, transition_profile_encoding_all_variants_vector
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes, average_trace_length

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Transition profile distance all variants, completed :: ")

    # Calculate distance for every variant, represented as a vector or a matrix
    if profile == 'matrix':
        encoding = transition_profile_encoding_all_variants_matrix(log)
        distance_per_variant_list = []
        if distance_measure == "cosine":
            for reference_var in encoding:
                progress.update()
                dist = sum(cosine(reference_var[row], compared_var[row]) for row in range(len(reference_var))
                           for compared_var in encoding)
                distance_per_variant_list.append(dist)
        if distance_measure == "euclidean":
            for reference_var in encoding:
                progress.update()
                dist = sum(euclidean(reference_var[row], compared_var[row]) for row in range(len(reference_var))
                           for compared_var in encoding)
                distance_per_variant_list.append(dist)

    elif profile == 'vector':
        encoding = transition_profile_encoding_all_variants_vector(log)
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
    else:
        print('No profile assigned!')

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_variant_list), sum(distance_per_variant_list) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1)) / average_trace_length(log), (sum(distance_per_variant_list) - total_number_of_trace_classes(log)) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1))


# Average distance per case
def transition_profile_distance_all_cases(log, profile, distance_measure):
    from encoding import transition_profile_encoding_all_traces_matrix, transition_profile_encoding_all_traces_vector
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces, average_trace_length

    # Set up progress bar
    no_trace = len(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Transition profile distance all cases, completed :: ")

    # Calculate distance for every case, represented as a vector or a matrix
    if profile == 'matrix':
        encoding = transition_profile_encoding_all_traces_matrix(log)
        distance_per_trace_list = []
        if distance_measure == "cosine":
            for reference_trace in encoding:
                progress.update()
                dist = sum(cosine(reference_trace[row], compared_trace[row]) for row in range(len(reference_trace))
                           for compared_trace in encoding)
                distance_per_trace_list.append(dist)
        if distance_measure == "euclidean":
            for reference_trace in encoding:
                progress.update()
                dist = sum(euclidean(reference_trace[row], compared_trace[row]) for row in range(len(reference_trace))
                           for compared_trace in encoding)
                distance_per_trace_list.append(dist)

    elif profile == 'vector':
        encoding = transition_profile_encoding_all_traces_vector(log)
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

    else:
        print('No profile assigned!')

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_trace_list), sum(distance_per_trace_list) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1)) / average_trace_length(log), (sum(distance_per_trace_list) - total_number_of_traces(log)) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1))


# METHODS FOR MARKOV CHAIN TRANSITION PROFILE

# Average distance per variant
def markov_chain_transition_profile_distance_all_variants(log, profile, distance_measure):
    from encoding import markov_chain_transition_profile_encoding_all_variants_matrix, \
        markov_chain_transition_profile_encoding_all_variants_vector, \
        markov_chain_transition_profile_encoding_all_variants_vector_mean
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes, average_trace_length

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Markov chain transition profile distance all variants, completed :: ")

    # Calculate distance for every variant, represented as a vector or a matrix
    if profile == 'matrix':
        encoding = markov_chain_transition_profile_encoding_all_variants_matrix(log)
        distance_per_variant_list = []
        if distance_measure == "cosine":
            for reference_var in encoding:
                progress.update()
                dist = sum(cosine(reference_var[row], compared_var[row]) for row in range(len(reference_var))
                           for compared_var in encoding)
                distance_per_variant_list.append(dist)
        if distance_measure == "euclidean":
            for reference_var in encoding:
                progress.update()
                dist = sum(euclidean(reference_var[row], compared_var[row]) for row in range(len(reference_var))
                           for compared_var in encoding)
                distance_per_variant_list.append(dist)

    elif profile == 'vector':
        encoding = markov_chain_transition_profile_encoding_all_variants_vector(log)
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

    elif profile == 'vector_mean':
        encoding = markov_chain_transition_profile_encoding_all_variants_vector_mean(log)
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
    else:
        print('No profile assigned!')

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_variant_list), sum(distance_per_variant_list) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1)) / average_trace_length(log), (sum(distance_per_variant_list) - total_number_of_trace_classes(log)) / (total_number_of_trace_classes(log) * (total_number_of_trace_classes(log) - 1))


# Average distance per case
def markov_chain_transition_profile_distance_all_cases(log, profile, distance_measure):
    from encoding import markov_chain_transition_profile_encoding_all_traces_matrix, \
        markov_chain_transition_profile_encoding_all_traces_vector, \
        markov_chain_transition_profile_encoding_all_traces_vector_mean
    from fastdist.fastdist import cosine, euclidean
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_traces, average_trace_length

    # Set up progress bar
    no_trace = len(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Markov chain transition profile distance all cases, completed :: ")

    # Calculate distance for every case, represented as a vector or a matrix
    if profile == 'matrix':
        encoding = markov_chain_transition_profile_encoding_all_traces_matrix(log)
        distance_per_trace_list = []
        if distance_measure == "cosine":
            for reference_trace in encoding:
                progress.update()
                dist = sum(cosine(reference_trace[row], compared_trace[row]) for row in range(len(reference_trace))
                           for compared_trace in encoding)
                distance_per_trace_list.append(dist)
        if distance_measure == "euclidean":
            for reference_trace in encoding:
                progress.update()
                dist = sum(euclidean(reference_trace[row], compared_trace[row]) for row in range(len(reference_trace))
                           for compared_trace in encoding)
                distance_per_trace_list.append(dist)

    elif profile == 'vector':
        encoding = markov_chain_transition_profile_encoding_all_traces_vector(log)
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

    elif profile == 'vector_mean':
        encoding = markov_chain_transition_profile_encoding_all_traces_vector_mean(log)
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
    else:
        print('No profile assigned!')

    # Close progress bar and return
    progress.close()
    del progress
    return np.asarray(distance_per_trace_list), sum(distance_per_trace_list) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1)) / average_trace_length(log), (sum(distance_per_trace_list) - total_number_of_traces(log)) / (total_number_of_traces(log) * (total_number_of_traces(log) - 1))


# Average distance per variant
def markov_chain_transition_profile_minimum_cosine_all_variants(log):
    from encoding import markov_chain_transition_profile_encoding_all_variants_vector
    from fastdist.fastdist import cosine
    from measures_extracted_from_literature.derived_from_linear_structures import total_number_of_trace_classes

    # Set up progress bar
    no_trace = total_number_of_trace_classes(log)  # count number of traces and setup progress bar
    progress = tqdm(total=no_trace, leave=False, desc="Markov chain transition profile distance all variants, completed :: ")

    # Calculate distance for every variant
    encoding = markov_chain_transition_profile_encoding_all_variants_vector(log)
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
