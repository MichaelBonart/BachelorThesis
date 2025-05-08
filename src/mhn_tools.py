import mhn
import numpy as np
import pandas as pd


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

