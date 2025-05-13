import mhn
import numpy as np
import pandas as pd

import EventDistanceMeasurer as edm


def copyMHN(a:mhn.model.cMHN):
    b=mhn.model.cMHN(np.ndarray.copy(a.log_theta), events=a.events)

    return b



def getLambdaSearchRange(data, steps:int=9):

    default_lambda= 1.0/len(data)

    lambda_min = 0.1 * default_lambda
    lambda_max = 1000 * default_lambda

    lambda_path: np.ndarray = np.exp(np.linspace(
                np.log(lambda_min + 1e-10), np.log(lambda_max + 1e-10), steps))
    
    return lambda_path


def eliminateZeroRows(dataset: pd.DataFrame):
    mask= dataset.sum(axis=1)
    return dataset[mask>0]




def getDistMatrix(dataset: pd.DataFrame, n_test_events=3, dist:edm.DIST=edm.DIST.OFFDIAG_L1_SYM, test_event_set=None):
    events=list(dataset.columns)[1:]
    if test_event_set is None:
        test_event_set=events[0:n_test_events]

    cluster_event_set=[e for e in events if e not in test_event_set]
    dist_measurer = edm.EventDistanceMeasurerCP(test_event_set, cluster_event_set)
    dist_measurer.load_data(dataset)
    dist_measurer.train_All_MHNs(do_cv=False)
    dist_measurer.compute_distance_matrix(dist_measure=dist)
    return dist_measurer._dist_mat
