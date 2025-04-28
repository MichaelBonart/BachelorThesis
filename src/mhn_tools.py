import mhn
import numpy as np


def copyMHN(a:mhn.model.cMHN):
    b=mhn.model.cMHN(np.ndarray.copy(a.log_theta), events=a.events)

    return b

