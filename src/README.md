# Implementations of bachelor thesis

This folder includes all implementations used in the bachelor thesis grouped by folders that correspond to the thesis' chapters.


'definition/': implementation of the newly proposed distance measure for cancer progression events (chapter 2)
'evaluation/': implementation of evaluation process (chapter 3)
'application/': application of the distance measure on real datasets of cancer patients (chapter 4)


Some more files that were generally useful across different directories:
'mhn_tools.py': useful methods for plotting/copying MHNs and more
'checkpoints_mbonart.py': automatic saving/loading of previously computed results

Additionally there are the following two subdirectories, which contain generated results:
'mbonart_checkpoints/': results of MHN optimizations that can be loaded again in the future
'result_plots': automaitcally generated plots (usually of MHNs of interest)