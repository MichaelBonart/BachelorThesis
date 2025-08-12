import mhn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from os.path import dirname, abspath




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

#filter out characters that won't work in filenames
def event_id(ev:str):
    return "".join(filter(lambda char: char not in "/() ?!", str(ev)))


def plotMHNgroup(mhns:List[mhn.model.cMHN],identifier="Group"):
    for m in mhns:
        m.plot(colorbar=False)
            
        plt.savefig(f"{dirname(abspath(__file__))}/result_plots/{identifier}_{'_'.join([event_id(ev) for ev in m.events])}")

