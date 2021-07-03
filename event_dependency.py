# METHODS FOR CALCULATING AND EVALUATING THE EVENT LOG'S EVENT DEPENDENCY MATRIX

def predecessor_event_density_dict(log):
    from general_methods import event_names
    from pm4py.objects.dfg.retrieval.log import native
    dfg_counts = native(log)
    events = sorted(event_names(log))
    predecessor_event_density_dict = {}
    for i in events:
        predecessor_count = 0
        predecessor_number = 0
        for j in events:
            if (j, i) in dfg_counts:
                predecessor_count += 1
                predecessor_number += dfg_counts[(j, i)]
        if predecessor_count > 0:
            predecessor_event_density_dict[i] = predecessor_number / predecessor_count
        else:
            predecessor_event_density_dict[i] = 0
    return predecessor_event_density_dict


def successor_event_density_dict(log):
    from general_methods import event_names
    from pm4py.objects.dfg.retrieval.log import native
    dfg_counts = native(log)
    events = sorted(event_names(log))
    successor_event_density_dict = {}
    for i in events:
        successor_count = 0
        successor_number = 0
        for j in events:
            if (i, j) in dfg_counts:
                successor_count += 1
                successor_number += dfg_counts[(i, j)]
        if successor_count > 0:
            successor_event_density_dict[i] = successor_number / successor_count
        else:
            successor_event_density_dict[i] = 0
    return successor_event_density_dict


def local_dependency(log, x, y, c=4):
    from pm4py.objects.dfg.retrieval.log import native
    from math import e
    dfg_counts = native(log)
    suc_dens_dict = successor_event_density_dict(log)
    pre_dens_dict = predecessor_event_density_dict(log)
    if (x, y) in dfg_counts:
        loc_dep = 1 - (1 / (1 + e**((dfg_counts[(x, y)] - suc_dens_dict[x]) * (c / suc_dens_dict[x]))) * 2) - \
                  (1 / (1 + e**((dfg_counts[(x, y)] - pre_dens_dict[y]) * (c / pre_dens_dict[y]))) * 2)
    else:
        loc_dep = 0
    return loc_dep


# Calculate event dependency matrix
def event_dependency_matrix(log, c=4):
    from general_methods import event_names
    import numpy as np
    import pandas as pd
    from pm4py.objects.dfg.retrieval.log import native
    from math import e

    # Prepare required data from log
    events = sorted(event_names(log))
    matrix_shape = len(events)
    event_dependency_matrix = [[0] * matrix_shape for i in range(matrix_shape)]
    dfg_counts = native(log)
    suc_dens_dict = successor_event_density_dict(log)
    pre_dens_dict = predecessor_event_density_dict(log)

    # Calculate event dependency matrix
    for k, i in enumerate(events):
        for l, j in enumerate(events):
            if i is not j:
                if (i, j) in dfg_counts:
                    loc_dep = 1 - (1 / ((1 + e**((dfg_counts[(i, j)] - suc_dens_dict[i]) * (c / suc_dens_dict[i]))) * 2)) - (1 / ((1 + e ** ((dfg_counts[(i, j)] - pre_dens_dict[j]) * (c / pre_dens_dict[j]))) * 2))
                    event_dependency_matrix[k][l] = loc_dep
                else:
                    event_dependency_matrix[k][l] = 0
            else:
                event_dependency_matrix[k][l] = 0

    # uncomment code to print matrix to excel file
    # with pd.ExcelWriter("02_Event_Dependency.xlsx") as writer:
    #     df = pd.DataFrame(np.asarray(event_dependency_matrix), columns=events, index=events)
    #     df.to_excel(writer, sheet_name='Matrix')

    return np.asarray(event_dependency_matrix)

