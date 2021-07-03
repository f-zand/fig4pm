# METHODS FOR DETECTING OUTLIERS IN UNI-VARIATE DATA

# Calculate upper and lower boundary breaches based on box-whisker-plot fences
def box_whisker_plot_evaluation(base_list):
    from numpy import quantile
    quantile_25, quantile_50, quantile_75 = quantile(base_list, [0.25, 0.5, 0.75])
    iqr = quantile_75 - quantile_25
    lower_bound = quantile_25 - (1.5 * iqr)
    upper_bound = quantile_75 + (1.5 * iqr)
    counter_upper_bound = 0
    for i in base_list:
        if i > upper_bound:
            counter_upper_bound += 1
    counter_lower_bound = 0
    for n in base_list:
        if n < lower_bound:
            counter_lower_bound += 1
    return counter_upper_bound, counter_lower_bound


# Calculate upper and lower boundary breaches based on median absolute deviation
def mad_evaluation(base_list):
    from statistics import median
    from scipy.stats import median_abs_deviation
    median = median(base_list)
    mad = median_abs_deviation(base_list)
    b = 1.4826          # value of b taken from: "Detecting outliers: Do not use standard deviation around the mean,
                        # use absolute deviation around the median" (Leys et al. 2013)
    lower_bound = median - (2.5 * (mad * b))
    upper_bound = median + (2.5 * (mad * b))
    counter_upper_bound = 0
    for i in base_list:
        if i > upper_bound:
            counter_upper_bound += 1
    counter_lower_bound = 0
    for n in base_list:
        if n < lower_bound:
            counter_lower_bound += 1
    return counter_upper_bound, counter_lower_bound


# Calculate upper and lower boundary breaches based on mean and standard deviation
def mean_stdv_evaluation(base_list):
    from statistics import mean, stdev
    mean = mean(base_list)
    stdv = stdev(base_list)
    lower_bound = mean - (2.5 * stdv)
    upper_bound = mean + (2.5 * stdv)
    counter_upper_bound = 0
    for i in base_list:
        if i > upper_bound:
            counter_upper_bound += 1
    counter_lower_bound = 0
    for n in base_list:
        if n < lower_bound:
            counter_lower_bound += 1
    return counter_upper_bound, counter_lower_bound


# UNSUPERVISED METHOD FOR DETECTING OUTLIERS IN MULTI-VARIATE DATA BASED ON EVENT LOG FEATURES


def outlier_detection_feature_based_unsupervised(log, model, contamination, distance=None, feature_corr=0.9):

    # Import general modules
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from general_methods import case_list
    from measures_extracted_from_literature.derived_from_linear_structures import minimum_trace_length, total_number_of_traces
    from profile_distances.trace_profile import trace_distance_all_cases
    from profile_distances.event_profile import event_profile_distance_all_cases
    from profile_distances.k_gram_profile import k_gram_distance_all_cases
    from profile_distances.degree_vector import degree_profile_distance_all_cases, out_degree_profile_distance_all_cases

    # Import models
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM

    # Fit the data to the model
    if contamination == None:
        return 'Contamination not defined!'
    if model == 'CBLOF':
        clf = CBLOF(contamination=contamination)
    elif model == 'HBOS':
        clf = HBOS(contamination=contamination)
    elif model == 'IForest':
        clf = IForest(contamination=contamination, behaviour='new')
    elif model == 'KNN':
        if distance == None:
            return 'No distance assigned!'
        else:
            metric = distance
        clf = KNN(metric=metric, contamination=contamination)
    elif model == 'LOF':
        if distance == None:
            return 'No distance assigned!'
        else:
            metric = distance
        clf = LOF(metric=metric, contamination=contamination)
    elif model == 'OCSVM':
        clf = OCSVM(contamination=contamination)
    else:
        return 'No model assigned!'

    # Create first columns of the dataframe
    case_num = [case.attributes["concept:name"] for case in log]
    traces = case_list(log)

    # Set up dataframe
    try:
        d = {'Case_Number': case_num, 'Case_Representation': traces}
        df = pd.DataFrame(data=d)
    except Exception:
        d = {'Case_Number': case_num}
        df = pd.DataFrame(data=d)

    # Add profile distances to dataframe
    df['Trace Distance'] = trace_distance_all_cases(log)[0]
    df['Event Profile Euclidean Distance'] = event_profile_distance_all_cases(log, 'euclidean')[0]
    df['Event Profile Cosine Distance'] = event_profile_distance_all_cases(log, 'cosine')[0]
    df['2 Gram Profile Euclidean Distance'] = k_gram_distance_all_cases(log, 2, 'euclidean')[0]
    df['2 Gram Profile Cosine Distance'] = k_gram_distance_all_cases(log, 2, 'cosine')[0]
    df['Degree Profile Euclidean Distance'] = degree_profile_distance_all_cases(log, 'euclidean')[0]
    df['Degree Profile Cosine Distance'] = degree_profile_distance_all_cases(log, 'cosine')[0]
    df['Out-Degree Profile Euclidean Distance'] = out_degree_profile_distance_all_cases(log, 'euclidean')[0]
    df['Out-Degree Profile Cosine Distance'] = out_degree_profile_distance_all_cases(log, 'cosine')[0]

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Trace Distance', 'Event Profile Euclidean Distance', 'Event Profile Cosine Distance',
        '2 Gram Profile Euclidean Distance', '2 Gram Profile Cosine Distance', 'Degree Profile Euclidean Distance',
        'Degree Profile Cosine Distance', 'Out-Degree Profile Euclidean Distance',
        'Out-Degree Profile Cosine Distance']] = \
        scaler.fit_transform(df[['Trace Distance', 'Event Profile Euclidean Distance', 'Event Profile Cosine Distance',
                                 '2 Gram Profile Euclidean Distance', '2 Gram Profile Cosine Distance',
                                 'Degree Profile Euclidean Distance', 'Degree Profile Cosine Distance',
                                 'Out-Degree Profile Euclidean Distance', 'Out-Degree Profile Cosine Distance']])

    # Reshape data
    X1 = df['Trace Distance'].values.reshape(-1, 1)
    X2 = df['Event Profile Euclidean Distance'].values.reshape(-1, 1)
    X3 = df['Event Profile Cosine Distance'].values.reshape(-1, 1)
    X4 = df['2 Gram Profile Euclidean Distance'].values.reshape(-1, 1)
    X5 = df['2 Gram Profile Cosine Distance'].values.reshape(-1, 1)
    X6 = df['Degree Profile Euclidean Distance'].values.reshape(-1, 1)
    X7 = df['Degree Profile Cosine Distance'].values.reshape(-1, 1)
    X8 = df['Out-Degree Profile Euclidean Distance'].values.reshape(-1, 1)
    X9 = df['Out-Degree Profile Cosine Distance'].values.reshape(-1, 1)

    # Profiles included in case underlying log allows usage
    if minimum_trace_length(log) == 2:
        X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9), axis=1)

    elif minimum_trace_length(log) == 3:
        df['3 Gram Profile Euclidean Distance'] = k_gram_distance_all_cases(log, 3, 'euclidean')[0]
        df['3 Gram Profile Cosine Distance'] = k_gram_distance_all_cases(log, 3, 'cosine')[0]

        # Scale data
        df[['3 Gram Profile Euclidean Distance', '3 Gram Profile Cosine Distance']] = \
            scaler.fit_transform(df[['3 Gram Profile Euclidean Distance', '3 Gram Profile Cosine Distance']])

        # Reshape data
        X10 = df['3 Gram Profile Euclidean Distance'].values.reshape(-1, 1)
        X11 = df['3 Gram Profile Cosine Distance'].values.reshape(-1, 1)
        X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11), axis=1)

    else:
        df['3 Gram Profile Euclidean Distance'] = k_gram_distance_all_cases(log, 3, 'euclidean')[0]
        df['3 Gram Profile Cosine Distance'] = k_gram_distance_all_cases(log, 3, 'cosine')[0]
        df['4 Gram Profile Euclidean Distance'] = k_gram_distance_all_cases(log, 4, 'euclidean')[0]
        df['4 Gram Profile Cosine Distance'] = k_gram_distance_all_cases(log, 4, 'cosine')[0]

        # Scale data
        df[['3 Gram Profile Euclidean Distance', '3 Gram Profile Cosine Distance', '4 Gram Profile Euclidean Distance',
            '4 Gram Profile Cosine Distance']] = \
            scaler.fit_transform(df[['3 Gram Profile Euclidean Distance', '3 Gram Profile Cosine Distance',
                                     '4 Gram Profile Euclidean Distance', '4 Gram Profile Cosine Distance']])

        # Reshape data
        X10 = df['3 Gram Profile Euclidean Distance'].values.reshape(-1, 1)
        X11 = df['3 Gram Profile Cosine Distance'].values.reshape(-1, 1)
        X12 = df['4 Gram Profile Euclidean Distance'].values.reshape(-1, 1)
        X13 = df['4 Gram Profile Cosine Distance'].values.reshape(-1, 1)
        X = np.concatenate((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13), axis=1)


    # Fit the data to the model
    clf.fit(X)

    # Predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1

    # Prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    regular_inliers = len(y_pred) - np.count_nonzero(y_pred)
    regular_outliers = np.count_nonzero(y_pred == 1)

    # Insert outlier tag into dataframe
    df['Score'] = scores_pred
    df['Outlier'] = y_pred

    # ---------------------------------------------------------------------------------------------------

    # Evaluate Feature Correlation
    try:
        dfx = df.drop(columns=['Case_Number', 'Case_Representation', 'Score', 'Outlier'])
    except KeyError:
        dfx = df.drop(columns=['Case_Number', 'Score', 'Outlier'])

    corr = dfx.corr()
    cor_matrix = corr.values.tolist()

    rows_to_drop = set()
    for row in range(len(cor_matrix)):
        if not cor_matrix[row][-1] == 'dropped':
            for column in range(row + 1, len(cor_matrix[row])):
                if not -feature_corr < cor_matrix[row][column] < feature_corr:
                    rows_to_drop.add(column)

    cols = list(rows_to_drop)
    dfx_2 = dfx.drop(dfx.columns[cols], axis=1)
    corr_2 = dfx_2.corr()

    l = []
    for i in range(len(dfx_2.columns)):
        l.append(dfx_2.iloc[:, i].values.reshape(-1, 1))

    Y = np.concatenate((l), axis=1)

    # Fit the data to the model
    clf.fit(Y)

    # Predict raw anomaly score
    scores_pred = clf.decision_function(Y) * -1

    # Prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(Y)
    reduced_inliers = len(y_pred) - np.count_nonzero(y_pred)
    reduced_outliers = np.count_nonzero(y_pred == 1)

    # Insert outlier tag into dataframe
    try:
        dfx_2['Score'] = scores_pred
        dfx_2['Outlier'] = y_pred
        dfx_2.insert(0, 'Case_Representation', traces)
        dfx_2.insert(0, 'Case_Number', case_num)
    except ValueError:
        dfx_2['Score'] = scores_pred
        dfx_2['Outlier'] = y_pred
        dfx_2.insert(0, 'Case_Number', case_num)

    # ----------------------------------------------------------------------------------------------------------

    # uncomment code to print feature evaluation to excel file
    # with pd.ExcelWriter("04_Feature_Distance_Evaluation.xlsx") as writer:
    #     df.to_excel(writer, sheet_name='Distance_Summary')
    #     dfx.to_excel(writer, sheet_name='Correlation_Features')
    #     corr.to_excel(writer, sheet_name='Correlation')
    #     corr_2.to_excel(writer, sheet_name='Correlation_2')
    #     dfx_2.to_excel(writer, sheet_name='Correlation_Features_New')

    return regular_outliers, regular_inliers, reduced_outliers, reduced_inliers
