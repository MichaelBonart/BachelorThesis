from enum import Enum
from typing import Dict
import mhn
import numpy as np
import pandas as pd
import cmhn_distances
import checkpoints_mbonart as cp
import mhn_tools as mytools
from pathlib import Path


class EventDistanceMeasurer:
    
    class DistMeasure(Enum):
         
        def __new__(cls, *args, **kwds):
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = value
            return obj

        def __init__(self, name:str, symmetrical:bool):
            self._name=name
            self._symmetrical=symmetrical
        
        TOTAL_EUCLID="Total euclid distance",True
        OFFDIAG_EUCLID="Offdiags' euclid distance", True

    global DIST
    DIST=DistMeasure

    def __init__(self, test_events, events):
        self._test_events = test_events
        self._events = events
        self._mhns:Dict[str, mhn.model.cMHN]={ev: None for ev in self._events}



    def load_data(self, data:pd.DataFrame):
        self._data=data

    def print_training_info(self, do_cv=True):
        mhn_test=mhn.optimizers.cMHNOptimizer()
        reduced_data=self._data[self._test_events]
        mhn_test.load_data_matrix(reduced_data)
        mhn_test.set_penalty(mhn.optimizers.cMHNOptimizer.Penalty.L1)
        searchrange=mytools.getLambdaSearchRange(reduced_data, steps=19)
        print(searchrange)

    def train_All_MHNs(self, measure_training_times:bool = False,do_cv=True, pick_1se=True):
        #TODO: compute init_theta and lambda universally on test_events only?
        mhn_test=mhn.optimizers.cMHNOptimizer()
        reduced_data=self._data[self._test_events]
        mhn_test.load_data_matrix(reduced_data)
        mhn_test.set_penalty(mhn.optimizers.cMHNOptimizer.Penalty.L1)
        if do_cv:
            searchrange=mytools.getLambdaSearchRange(reduced_data, steps=19)
            print(searchrange)
            self._lam_test, scores= mhn_test.lambda_from_cv(lambda_vector=searchrange,pick_1se=pick_1se, return_lambda_scores=True)
            print(self._lam_test)
            print(scores)
        else:
            self._lam_test = 1/len(self._data)
        
        mhn_test.train(self._lam_test)
        self._init_theta = np.pad(mhn_test.result.log_theta, ((0,1),(0,1)))

        #train individual MHNs
        for ev in self._events:
            print(f"training MHN for event {ev}")
            self.__train_MHN(ev)


    def __train_MHN(self, ev:str):
        mhn_opt=mhn.optimizers.cMHNOptimizer()
        reduced_data=self._data[self._test_events + [ev]]
        mhn_opt.load_data_matrix(reduced_data)
        mhn_opt.set_init_theta(self._init_theta)
        mhn_opt.set_penalty(mhn.optimizers.cMHNOptimizer.Penalty.L1)
        #lam= mhn_opt.lambda_from_cv(nfolds=3,steps=5)
        lam=self._lam_test
        mhn_opt.train(lam)
        self._mhns[ev]=mhn_opt.result

    def getDistMeasureFunc(self, dist_measure:DistMeasure):
        print(f"get func for {dist_measure}")
        match dist_measure:
            case self.DistMeasure.TOTAL_EUCLID:
                return cmhn_distances.total_euclid_dist
            case self.DistMeasure.OFFDIAG_EUCLID:
                return cmhn_distances.euclid_dist_offdiag
        
        print("Nothing matched")
        return -1

    def compute_distance_matrix(self, dist_measure:DistMeasure):
        n=len(self._events)
        dist_func=self.getDistMeasureFunc(dist_measure)
        print(f"Distance function: {dist_func}")
        self._dist_mat=pd.DataFrame(np.asarray(
            [[dist_func(self._mhns[p1], self._mhns[p2])
                for p2 in self._events] for p1 in self._events]
            ), index=self._events, columns=self._events)
        return self._dist_mat
    

    #TODO: save and load dataframe (training data) too??

    #filter out characters that won't work in filenames
    def event_id(self, ev:str):
        return "".join(filter(lambda char: char not in "/() ?!", str(ev)))

    def saveto(self, dir:str):
        #save all computed data in directory 'dir'
        Path(dir).mkdir(parents=False, exist_ok=True)
        for ev in self._events:
            self._mhns[ev].save(filename=f"{dir}/mhn_{self.event_id(ev)}")



    def loadfrom(self, dir:str):
        #load all data stored in directory 'dir'
        for ev in self._events:
            self._mhns[ev] = mhn.model.cMHN.load(filename=f"{dir}/mhn_{self.event_id(ev)}", events=self._test_events + [ev])







#child class of EventDistanceMeasurer that implements automatic saving/loading of computation heavy results in the 'checkpoints' directory
class EventDistanceMeasurerCP(EventDistanceMeasurer):

    def train_All_MHNs(self, measure_training_times = False, identifier="", **kwargs):
        hash= pd.util.hash_pandas_object(self._data, index=True).sum() + len(self._test_events)
        print(hash.hex())
        hashstr=hash.hex()[4:-4]
        dirname=f"edm_{identifier}{hashstr}"
        print(f"Directory for storage is {dirname}")

        if not cp.is_dir_already_computed(dirname):
            super().train_All_MHNs(measure_training_times, **kwargs)
            self.saveto(f"{cp.CHECKPOINT_DIR}/{dirname}")
        else:
            self.loadfrom(f"{cp.CHECKPOINT_DIR}/{dirname}")
    