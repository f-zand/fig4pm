# METHODS FOR CALCULATING AND EVALUATING THE EVENT LOG'S EVENT SPATIAL PROXIMITY MATRIX

def spatial_proximity_matrix(log):
    from general_methods import case_list, event_names
    import numpy as np
    import pandas as pd

    # Prepare required data from log
    events = sorted(event_names(log))
    cases = case_list(log)
    matrix_shape = len(events)
    spatial_proximity_matrix = [[0] * matrix_shape for i in range(matrix_shape)]

    # Calculate spatial proximity matrix
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            spatial_proximity = 0
            if events[i] is not events[j]:
                case_count = 0
                for case in cases:
                    i_indices = []
                    j_indices = []
                    distances = []
                    if all(x in case for x in [events[i], events[j]]):
                        case_count += 1
                        for index, event in enumerate(case):
                            if event == events[i]:
                                i_indices.append(index)
                            if event == events[j]:
                                j_indices.append(index)
                        for ind_i in i_indices:
                            for ind_j in j_indices:
                                distances.append(abs(ind_i - ind_j))
                        spatial_proximity += (1 - (min(distances) / len(case)))
                if case_count == 0:
                    spatial_proximity_matrix[i][j] = 0
                    spatial_proximity_matrix[j][i] = 0
                else:
                    spatial_proximity_matrix[i][j] += spatial_proximity / case_count
                    spatial_proximity_matrix[j][i] += spatial_proximity / case_count
            else:
                spatial_proximity_matrix[i][j] = 0

    # uncomment code to print matrix to excel file
    # with pd.ExcelWriter("03_Spatial_Proximity.xlsx") as writer:
    #     df = pd.DataFrame(np.asarray(spatial_proximity_matrix), columns=events, index=events)
    #     df.to_excel(writer, sheet_name='Matrix')

    return np.asarray(spatial_proximity_matrix)
