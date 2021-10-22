###### This repository was build by Peter D. from University of Mannheim as a part of a master thesis study.
#
## Where to start?
You can load your `.csv` or `.xes` event log by making changes in `event_log_assessment.py` file.

## Event_Log_Assessment
This repository contains a tool for assessing a variety of event log characteristics via log-based features. The features are utilized for the recommendation of specific pre-discovery actions while also allowing for sub-log comparison.

An overview of the files contained in this repository as well as their content is given in the following:
   #### features extracted from the literature 
   The folder includes two files, which contain the implementation of features extracted from the literature. The two files distinguish the features by means of their grounding on features derived from linear structures or non-linear structures. The features ordering follows the procedure and descriptions of the thesis.

   #### self_developed_features
   The folder includes two files, which contain the implementation of self-developed features. The two files distinguish the features by means of their grounding on features derived from linear structures or non-linear structures. The features ordering follows the procedure and descriptions of the thesis.

   #### evaluation
   The python file included in this folder contains the method used for the features evaluation. The method creates an excel file depicting all features and their calculated value. Further the folder contains a subfolder including the evaluation results in two excel files. The 'outlier_detection_measure_results' file exhibits the results obtained during the pre-evaluation of the comprehensive outlier detection algorithm. Testing was done on the complete feature vector and the reduced feature vector, i.e. the vector after dropping dimensions that showed a correlation of more than 90% to another one. Therefore, the included tables show results for the 'initial' and the 'reduced' feature vector. The 'measures_evaluation_results' file depicts the results obtained by applying the aformentioned evaluation algorithm to 49 synthetic and real-life event logs. In case a certain feature's value is not depicted in the results table, the feature's computation time was too extensive. The file further contains the correlation analysis conducted for the complete set of features.

   #### event_log_assessment.py
   This file includes the final method developed throughout the thesis. Applied to an event log, it returns all features and their corresponding values, following the structure that was detected as the best suited one during the evaluation stage.

   #### encoding.py
   This file contains methods allowing to encode an event log to a certain profile. The possible profiles comprehend the event profile, k-gram profile, different transition profiles and degree profiles for all variants or cases of an event log.

   #### profile_distances
   This folder contains five files including methods that allow to calculate distances or similarities for all variants or cases in the event log. On the one hand, the methods enable to calculate the average distance or similarity between all variants or traces in the event log. On the other hand, they allow to create a list for each variant or trace containing its aggregated distance or similarity to all other variants or traces in the log.

   #### general_methods.py
   This file contains general methods used for the implementation of the various features. The methods range from the calculation of case- and variant-lists, over adjacency matrices to retrieving a list of self-loops and repetitions contained in an event log.

   #### outlier_detection.py
   This file contains the outlier detection algorithms used throughout the thesis. These are, on the one hand, the implementation of the boxplot interquartile range approach, and the comprehensive unsupervised outlier detection approach on the other hand.

   #### graph_creation.py
   This file contains methods creating undirected or directed graphs based on an event log.

   #### event_dependency.py
   This file contains methods allowing to derive an event dependency matrix from a given event log.

   #### spatial_proximity.py
   This file contains methods allowing to derive a spatialy proximity matrix from a given event log.

