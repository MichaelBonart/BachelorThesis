# EVALUATION

The files in this directory outline the evaluation process described in chapter 3 of the bachelor thesis.

-'ground_truth_generation.ipynb':   generation of a ground truth MHN which is used in 'noise_comparison.ipynb'
-'noise_comparison.ipynb':          main evaluation with adding of noise and perturbation, as well as scoring with adjusted rand score.
-'ClusterableMhnGenerator.py':      splitting up an MHN into chosen clusters   
-'RandomMHNGenerator.py':           applying random noise to an MHN, do training iterations to ensure good convergence properties.
-'cv_results_investigation':        comparison of cross-validation results for MHNs of different sizes 2-8