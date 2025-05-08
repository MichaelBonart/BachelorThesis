import mhn
import numpy as np


def total_euclid_dist(mhn1: mhn.model.cMHN, mhn2: mhn.model.cMHN):
    return np.sum((mhn1.log_theta-mhn2.log_theta)**2)



def euclid_dist_offdiag(mhn1: mhn.model.cMHN, mhn2: mhn.model.cMHN):
    diff=(mhn1.log_theta-mhn2.log_theta)
    np.fill_diagonal(diff,0)
    return np.sum(diff**2)

def offdiag_l1(mhn1: mhn.model.cMHN, mhn2: mhn.model.cMHN):
    diff=np.abs(mhn1.log_theta-mhn2.log_theta)
    np.fill_diagonal(diff,0)
    return np.sum(diff)


def offdiag_l1_sym(mhn1: mhn.model.cMHN, mhn2: mhn.model.cMHN):
    sym1=mhn1.log_theta + np.transpose(mhn1.log_theta)
    sym2=mhn2.log_theta + np.transpose(mhn2.log_theta)

    diff=np.abs(sym1-sym2)/2
    np.fill_diagonal(diff,0)
    return np.sum(diff)
