# METHODS USED FOR ENCODING THE LOG'S TRACES AND VARIANTS
import numpy as np

# EVENT PROFILE ENCODING - 1 VECTOR PER TRACE:

# Encoding of all traces in the log:
def event_profile_encoding_all_traces(log):
    from general_methods import event_names

    event_names = sorted(event_names(log))
    trace_encoding_event_profile = []
    for case in log:
        trace_encoding = dict.fromkeys(event_names, 0)
        for event in case:
            trace_encoding[event["concept:name"]] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Encoding of all variants in the log:
def event_profile_encoding_all_variants(log):
    from general_methods import variant_list
    from general_methods import event_names

    var_list = variant_list(log)
    event_names = sorted(event_names(log))
    variant_encoding_event_profile = []
    for variant in var_list:
        variant_encoding = dict.fromkeys(event_names, 0)
        for event in variant:
            variant_encoding[event] += 1
        variant_encoding_event_profile.append([*variant_encoding.values()])
    return np.asarray(variant_encoding_event_profile)


# K-GRAM ENCODING - 1 VECTOR PER TRACE:

# Encoding of all traces in the log:
def k_gram_encoding_all_traces(log, k):
    from general_methods import variant_list, case_list

    variant_list = variant_list(log)

    # create k-gram list
    k_gram_list = []
    for var in variant_list:
        for i in range(len(var) - k + 1):
            k_gram = ''.join(var[i:(i+k)])
            if k_gram not in k_gram_list:
                k_gram_list.append(k_gram)

    # create k-gram encoding
    k_gram_enc = []
    case_list = case_list(log)
    for case in case_list:
        case_k_gram_enc = []
        case_k_gram_list = [''.join(case[i:(i+k)]) for i in range(len(case) - k + 1)]
        for i in k_gram_list:
            counter = 0
            for j in case_k_gram_list:
                if i == j:
                    counter += 1
            case_k_gram_enc.append(counter)
        k_gram_enc.append(case_k_gram_enc)
    return np.asarray(k_gram_enc)


# Encoding of all variants in the log:
def k_gram_encoding_all_variants(log, k):
    from general_methods import variant_list

    var_list = variant_list(log)

    # create k-gram list
    k_gram_list = []
    for var in var_list:
        for i in range(len(var) - k + 1):
            k_gram = ''.join(var[i:(i+k)])
            if k_gram not in k_gram_list:
                k_gram_list.append(k_gram)

    # create k-gram encoding
    k_gram_enc = []
    for var in var_list:
        var_k_gram_enc = []
        var_k_gram_list = [''.join(var[i:(i+k)]) for i in range(len(var) - k + 1)]
        for i in k_gram_list:
            counter = 0
            for j in var_k_gram_list:
                if i == j:
                    counter += 1
            var_k_gram_enc.append(counter)
        k_gram_enc.append(var_k_gram_enc)
    return np.asarray(k_gram_enc)


# RANK PROFILE ENCODING - 1 VECTOR PER TRACE:

# Encoding of all traces in the log:
def rank_profile_encoding_all_traces(log):
    from general_methods import case_list, ranking_dict

    case_rep = case_list(log)
    ranking_dict = ranking_dict(log)
    case_rank_encoding_list = []
    for case in case_rep:
        case_rank_encoding = []
        for event in case:
            case_rank_encoding.append(ranking_dict[event])
        case_rank_encoding_list.append(case_rank_encoding)
    return np.asarray(case_rank_encoding_list)


# Encoding of all variants in the log:
def rank_profile_encoding_all_variants(log):
    from general_methods import variant_list, ranking_dict

    var_list = variant_list(log)
    ranking_dict = ranking_dict(log)
    variant_rank_encoding_list = []
    for variant in var_list:
        variant_rank_encoding = []
        for event in variant:
            variant_rank_encoding.append(ranking_dict[event])
        variant_rank_encoding_list.append(variant_rank_encoding)
    return np.asarray(variant_rank_encoding_list)


# TRANSITION PROFILE ENCODING - 1 SQUARE MATRIX PER TRACE:

# Encoding of all traces in the log - count +1 for each transition:
def transition_profile_encoding_all_traces_matrix(log):
    from general_methods import event_names

    case_rank_encoding_list = rank_profile_encoding_all_traces(log)
    matrix_shape = len(event_names(log))
    event_transition_profile = []
    for case in case_rank_encoding_list:
        initial_matrix = np.zeros(shape=(matrix_shape, matrix_shape))
        for (i, j) in zip(case, case[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        event_transition_profile.append(initial_matrix)
    return np.asarray(event_transition_profile)


# Encoding of all variants in the log - count +1 for each transition:
def transition_profile_encoding_all_variants_matrix(log):
    from general_methods import event_names

    variant_rank_encoding_list = rank_profile_encoding_all_variants(log)
    matrix_shape = len(event_names(log))
    event_transition_profile = []
    for var in variant_rank_encoding_list:
        initial_matrix = np.zeros(shape=(matrix_shape, matrix_shape))
        for (i, j) in zip(var, var[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        event_transition_profile.append(initial_matrix)
    return np.asarray(event_transition_profile)


# Markov Chain Transition Profile Encoding of all traces in the log - probability for each transition:
def markov_chain_transition_profile_encoding_all_traces_matrix(log):
    from general_methods import event_names

    case_rank_encoding_list = rank_profile_encoding_all_traces(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for case in case_rank_encoding_list:
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(case, case[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
        event_transition_profile.append(initial_matrix)
    return np.asarray(event_transition_profile)


# Markov Chain Transition Profile Encoding of all variants in the log - probability for each transition:
def markov_chain_transition_profile_encoding_all_variants_matrix(log):
    from general_methods import event_names

    variant_rank_encoding_list = rank_profile_encoding_all_variants(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for var in variant_rank_encoding_list:
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(var, var[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
        event_transition_profile.append(initial_matrix)
    return np.asarray(event_transition_profile)


# TRANSITION PROFILE ENCODING - 1 VECTOR PER TRACE:

# Encoding of all traces in the log - count +1 for each transition:
def transition_profile_encoding_all_traces_vector(log):
    from general_methods import event_names

    case_rank_encoding_list = rank_profile_encoding_all_traces(log)
    matrix_shape = len(event_names(log))
    event_transition_profile = []
    for case in case_rank_encoding_list:
        initial_matrix = np.zeros(shape=(matrix_shape, matrix_shape))
        for (i, j) in zip(case, case[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        event_transition_profile.append(initial_matrix)
    return np.reshape(event_transition_profile, newshape=(len(log), matrix_shape**2))


# Encoding of all variants in the log - count +1 for each transition:
def transition_profile_encoding_all_variants_vector(log):
    from general_methods import event_names

    variant_rank_encoding_list = rank_profile_encoding_all_variants(log)
    matrix_shape = len(event_names(log))
    event_transition_profile = []
    for var in variant_rank_encoding_list:
        initial_matrix = np.zeros(shape=(matrix_shape, matrix_shape))
        for (i, j) in zip(var, var[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        event_transition_profile.append(initial_matrix)
    return np.reshape(event_transition_profile, newshape=(len(variant_rank_encoding_list), matrix_shape**2))


# Markov Chain Transition Profile Encoding of all traces in the log - probability for each transition:
def markov_chain_transition_profile_encoding_all_traces_vector(log):
    from general_methods import event_names

    case_rank_encoding_list = rank_profile_encoding_all_traces(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for case in case_rank_encoding_list:
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(case, case[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
        event_transition_profile.append(initial_matrix)
    return np.reshape(event_transition_profile, newshape=(len(log), matrix_shape**2))


# Markov Chain Transition Profile Encoding of all variants in the log - probability for each transition:
def markov_chain_transition_profile_encoding_all_variants_vector(log):
    from general_methods import event_names

    variant_rank_encoding_list = rank_profile_encoding_all_variants(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for var in variant_rank_encoding_list:
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(var, var[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
        event_transition_profile.append(initial_matrix)
    return np.reshape(event_transition_profile, newshape=(len(variant_rank_encoding_list), matrix_shape**2))


# Markov Chain Transition Profile Encoding of all traces in the log - mean probability for each transition:
def markov_chain_transition_profile_encoding_all_traces_vector_mean(log):
    from general_methods import event_names

    case_rank_encoding_list = rank_profile_encoding_all_traces(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for case in case_rank_encoding_list:
        case_rep = []
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(case, case[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
            case_rep.append(np.mean(row))
        event_transition_profile.append(case_rep)
    return np.asarray(event_transition_profile)


# Markov Chain Transition Profile Encoding of all variants in the log - mean probability for each transition:
def markov_chain_transition_profile_encoding_all_variants_vector_mean(log):
    from general_methods import event_names

    variant_rank_encoding_list = rank_profile_encoding_all_variants(log)
    event_transition_profile = []
    matrix_shape = len(event_names(log))
    for var in variant_rank_encoding_list:
        var_rep = []
        initial_matrix = np.zeros(shape=[matrix_shape, matrix_shape])
        for (i, j) in zip(var, var[1:]):  # builds pairs of ranks directly following each other, e.g [(1,2), (2,3)]
            initial_matrix[i][j] += 1
        for row in initial_matrix:
            n = sum(row)
            if n > 0:
                row[:] = [(f / n) for f in row]
            var_rep.append(np.mean(row))
        event_transition_profile.append(var_rep)
    return np.asarray(event_transition_profile)


# UNDIRECTED GRAPH ENCODING - 1 VECTOR PER TRACE:

# Undirected Graph Degree Vector Encoding for all traces in the log
def undirected_graph_degree_vector_encoding_all_traces(log):
    from general_methods import case_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    cases = case_list(log)
    trace_encoding_event_profile = []
    for case in cases:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(case, case[1:]))
        for event in event_names:
            for elem in dfg:
                if event in elem:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Undirected Graph Degree Vector Encoding for all variants in the log
def undirected_graph_degree_vector_encoding_all_variants(log):
    from general_methods import variant_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    variants = variant_list(log)
    trace_encoding_event_profile = []
    for var in variants:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(var, var[1:]))
        for event in event_names:
            for elem in dfg:
                if event in elem:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Undirected Graph Outgoing Vector Encoding for all traces in the log
def undirected_graph_outgoing_vector_encoding_all_traces(log):
    from general_methods import case_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    cases = case_list(log)
    trace_encoding_event_profile = []
    for case in cases:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(case, case[1:]))
        for event in event_names:
            for elem in dfg:
                if event == elem[0]:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Undirected Graph Outgoing Vector Encoding for all variants in the log
def undirected_graph_outgoing_vector_encoding_all_variants(log):
    from general_methods import variant_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    variants = variant_list(log)
    trace_encoding_event_profile = []
    for var in variants:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(var, var[1:]))
        for event in event_names:
            for elem in dfg:
                if event == elem[0]:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Undirected Graph Incoming Vector Encoding for all traces in the log
def undirected_graph_incoming_vector_encoding_all_traces(log):
    from general_methods import case_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    cases = case_list(log)
    trace_encoding_event_profile = []
    for case in cases:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(case, case[1:]))
        for event in event_names:
            for elem in dfg:
                if event == elem[1]:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)


# Undirected Graph Incoming Vector Encoding for all variants in the log
def undirected_graph_incoming_vector_encoding_all_variants(log):
    from general_methods import variant_list
    from general_methods import event_names
    import numpy as np

    event_names = sorted(event_names(log))
    variants = variant_list(log)
    trace_encoding_event_profile = []
    for var in variants:
        trace_encoding = dict.fromkeys(event_names, 0)
        dfg = list(zip(var, var[1:]))
        for event in event_names:
            for elem in dfg:
                if event == elem[1]:
                    trace_encoding[event] += 1
        trace_encoding_event_profile.append([*trace_encoding.values()])
    return np.asarray(trace_encoding_event_profile)

