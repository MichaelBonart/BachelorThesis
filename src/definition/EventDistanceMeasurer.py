import __init__
from enum import Enum
from typing import Dict
import mhn
import numpy as np
import pandas as pd
import checkpoints_mbonart as cp
from definition import cmhn_distances
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
        OFFDIAG_EUCLID="Offdiagonals' euclid distance", True
        OFFDIAG_L1 = "Offdiagonals' L1 distance", True
        OFFDIAG_L1_SYM = "Symmetrized offdiagonals' L1 distance", True


    global DIST
    DIST=DistMeasure

    def __init__(self, test_events, events):
        self._test_events = test_events
        self._events = events
        self._mhns:Dict[str, mhn.model.cMHN]={ev: None for ev in self._events}
        self._null_mhn = None



    def load_data(self, data:pd.DataFrame):
        self._data=data


    def train_All_MHNs(self, do_cv=False, pick_1se=False):
        mhn_test=mhn.optimizers.cMHNOptimizer()
        reduced_data=self._data[self._test_events]

        mhn_test.load_data_matrix(reduced_data)
        mhn_test.set_penalty(mhn.optimizers.cMHNOptimizer.Penalty.L1)

        if do_cv:
            searchrange=mytools.getLambdaSearchRange(reduced_data, steps=19)
            self._lam_test = mhn_test.lambda_from_cv(lambda_vector=searchrange,pick_1se=pick_1se)
        else:
            self._lam_test = 1/len(reduced_data)
        
        mhn_test.train(self._lam_test)
        self._null_mhn = mhn.model.cMHN(np.pad(mhn_test.result.log_theta, ((0,1),(0,1))), events=self._test_events +['null'])

        #train individual MHNs
        for ev in self._events:
            print(f"training MHN for event {ev}")
            self.__train_MHN(ev)


    def __train_MHN(self, ev:str):
        mhn_opt=mhn.optimizers.cMHNOptimizer()
        reduced_data=self._data[self._test_events + [ev]]
        mhn_opt.load_data_matrix(reduced_data)
        mhn_opt.set_penalty(mhn.optimizers.cMHNOptimizer.Penalty.L1)
        lam=self._lam_test
        mhn_opt.train(lam)
        self._mhns[ev]=mhn_opt.result

    def getDistMeasureFunc(self, dist_measure:DistMeasure):
        match dist_measure:
            case self.DistMeasure.TOTAL_EUCLID:
                return cmhn_distances.total_euclid_dist
            case self.DistMeasure.OFFDIAG_EUCLID:
                return cmhn_distances.euclid_dist_offdiag
            case self.DistMeasure.OFFDIAG_L1:
                return cmhn_distances.offdiag_l1
            case self.DistMeasure.OFFDIAG_L1_SYM:
                return cmhn_distances.offdiag_l1_sym
        
        raise ValueError(f"Error: No distance measure function matched {dist_measure}")
        return -1

    def compute_distance_matrix(self, dist_measure:DistMeasure):
        if self._null_mhn is None:
            raise ValueError("Error: compute_distance_matrix requires call of 'train_All_MHNs' beforehand.")

        dist_func=self.getDistMeasureFunc(dist_measure)
        self._dist_mat=pd.DataFrame(np.asarray(
            [[dist_func(self._mhns[p1], self._mhns[p2])
                for p2 in self._events] for p1 in self._events]
            ), index=self._events, columns=self._events)
        return self._dist_mat
    

    #make dist measurer's domain include the test events
    #allows for easier combination of the distance matrices for dist measurers with different test event sets
    def extend_event_domain(self):
        for ev in self._test_events:
            self._mhns[ev]= self._null_mhn

        self._events.extend(self._test_events)
    

    #save all computed data in directory 'dir'
    def saveto(self, dir:str):
        Path(dir).mkdir(parents=True, exist_ok=True)
        for ev in self._events:
            self._mhns[ev].save(filename=f"{dir}/mhn_{mytools.event_id(ev)}.csv")

        self._null_mhn.save(filename=f"{dir}/mhn_empty.csv")



    #load all data stored in directory 'dir'
    def loadfrom(self, dir:str):
        for ev in self._events:
            try:
                self._mhns[ev] = mhn.model.cMHN.load(filename=f"{dir}/mhn_{mytools.event_id(ev)}.csv", events=self._test_events + [ev])
            
            except FileNotFoundError:   #TEMPORARY: access old files with no '.csv' extension
                self._mhns[ev] = mhn.model.cMHN.load(filename=f"{dir}/mhn_{mytools.event_id(ev)}", events=self._test_events + [ev])


        try:
            self._null_mhn = mhn.model.cMHN.load(filename=f"{dir}/mhn_empty.csv", events=self._test_events + ['null'])
        except FileNotFoundError:
            print("No init theta available in save files. Event domain extension only available after recomputing")




def getDistMeasurer(dataset: pd.DataFrame,events=None, n_test_events=3, dist:EventDistanceMeasurer.DistMeasure=EventDistanceMeasurer.DistMeasure.OFFDIAG_L1_SYM, test_event_set=None, extended_event_domain=False, identifier='')->EventDistanceMeasurer:
    
    if events is None:
        events=list(dataset.columns)
        
    if test_event_set is None:
        test_event_set=events[0:n_test_events]

    cluster_event_set=[e for e in events if e not in test_event_set]
    dist_measurer = EventDistanceMeasurerCP(test_event_set, cluster_event_set)
    dist_measurer.load_data(dataset)
    dist_measurer.train_All_MHNs(identifier=identifier)
    if extended_event_domain: 
        dist_measurer.extend_event_domain()
    dist_measurer.compute_distance_matrix(dist_measure=dist)
    return dist_measurer







#child class of EventDistanceMeasurer that implements automatic saving/loading of computation heavy results in the 'checkpoints' directory
#save file directory is determined by hash of dataset and the test events
class EventDistanceMeasurerCP(EventDistanceMeasurer):

    def train_All_MHNs(self, measure_training_times = False, identifier="", **kwargs):
        hash= pd.util.hash_pandas_object(self._data, index=True).sum() + len(self._test_events)
        hashstr=hash.hex()[4:-4]
        dirname=f"edm_{identifier}{hashstr}"

        dirname+=f"/{mytools.event_id('_'.join(self._test_events))}"
        print(f"Directory for storage is {dirname}")

        if not cp.is_dir_already_computed(dirname):
            super().train_All_MHNs(measure_training_times, **kwargs)
            self.saveto(cp.CHECKPOINT_DIR/dirname)
        else:
            self.loadfrom(cp.CHECKPOINT_DIR/dirname)
    