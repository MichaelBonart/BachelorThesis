import itertools
from typing import Dict, List
import mhn
import numpy as np
import string
import mhn_tools
from enum import Enum


"""
This class generates large MHNs for which a known clustering of events is already given.
Use this for evaluation of the clustering algorithm.
The most important methods, that should be called in this order are:
--setClusterMHN(mhn)
--split(clusterSizes)
--getMHN() -> resulting MHN


Further manipulation of the MHN is possible with the other methods.
"""
class ClusterableMhnGenerator:

    class CMG_Step(Enum):
        CLUSTERED=0,
        SPLIT=1,
        CONFUSCATED=2

    class CMG_Domain(Enum):
        TEST=0,
        CLUSTER=1,
        TOTAL=2

    global STP,DOM
    STP = CMG_Step
    DOM = CMG_Domain

    def __init__(self, test_events:list[str], event_clusters:list[str]):
        self.__initBlankState(test_events,event_clusters)
    
    def __init__(self, test_event_count:int, clusterCount:int):
        test_events = [f'T{i}' for i in range(test_event_count)]
        event_clusters = list(string.ascii_uppercase)[:clusterCount]
        self.__initBlankState(test_events,event_clusters)

    
    def __initBlankState(self, test_events:list[str], event_clusters:list[str]):
        self.events={}
        self.events[STP.CLUSTERED] = {DOM.TEST: test_events, DOM.CLUSTER: event_clusters}

        self.event_count = {}
        self._mhn : Dict[ClusterableMhnGenerator.CMG_Step, mhn.model.cMHN]= {}
        self.__complete_Initilisation_for_CMG_Step(STP.CLUSTERED)

    def __complete_Initilisation_for_CMG_Step(self, cmg_step: CMG_Step):
        #store total events
        self.events[cmg_step][DOM.TOTAL] = (self.events[cmg_step][DOM.TEST] + self.events[cmg_step][DOM.CLUSTER])
        #store all event counts
        self.event_count[cmg_step] = {key: len(events) for key,events in self.events[cmg_step].items()} 
        #setup zero-matrix as logtheta
        ec =self.event_count[cmg_step][DOM.TOTAL]
        self._mhn[cmg_step] = mhn.model.cMHN(np.zeros((ec,ec)), self.events[cmg_step][DOM.TOTAL])
        self._cStep= cmg_step


    def setClusterMHN(self, cluster_mhn:mhn.model.cMHN):
        self._mhn[STP.CLUSTERED].log_theta=cluster_mhn.log_theta
        return self
    
    def randomizeBaseRates(self):
        np.fill_diagonal(self._mhn[self._cStep].log_theta,np.random.pareto(3,self.event_count[self._cStep][DOM.TOTAL])-2.2)
        return self

    

    def splitClusters(self, clusterSizes:np.ndarray):
        self.events[STP.SPLIT]={DOM.TEST : self.events[STP.CLUSTERED][DOM.TEST]}
        split_cluster_names=[[f'{cluster}_{j}'\
                              for j in range(clusterSizes[i])]\
                                  for i,cluster in enumerate(self.events[STP.CLUSTERED][DOM.CLUSTER])]
        self.events[STP.SPLIT][DOM.CLUSTER]=\
            list(itertools.chain.from_iterable(split_cluster_names))
        
        self.__complete_Initilisation_for_CMG_Step(STP.SPLIT)

        #TODO: copy values from cluster MHN to split MHN
        repetitions=([1]*self.event_count[STP.CLUSTERED][DOM.TEST])+ clusterSizes
        x_p=0
        for nx,repetitionX in enumerate(repetitions):
            y_p=0
            for ny,repetitionY in enumerate(repetitions):
                for i in range(repetitionX):
                    for j in range(repetitionY):
                        if(x_p+i==y_p+j or nx!=ny):
                            self._mhn[STP.SPLIT].log_theta[x_p+i,y_p+j]=self._mhn[STP.CLUSTERED].log_theta[nx,ny]
                y_p+=repetitionY
            x_p+=repetitionX



        return self
    

    def getDomainMask(self, domain:CMG_Domain):
        mask = np.zeros(shape=( self.event_count[self._cStep][DOM.TOTAL]))
        match domain:
            case DOM.TEST: 
                mask[:self.event_count[self._cStep][DOM.TEST]]=1
                return mask
            
            case DOM.CLUSTER:
                mask[self.event_count[self._cStep][DOM.TEST]+1 : ]=1
                return mask
            
            case DOM.TOTAL:
                mask[:]=1
                return mask


    def addNoiseOffDiags(self, amplitude:float, domain=DOM.TOTAL, col_domain=None):
        if col_domain is None:
            col_domain=domain

        n=self.event_count[self._cStep][DOM.TOTAL]
        mask2d = np.outer(self.getDomainMask(domain), self.getDomainMask(col_domain))   #shape of nxn
        np.fill_diagonal(mask2d, 0)
        noise = np.random.normal(0, amplitude, size=(n,n))* mask2d

        self._mhn[self._cStep].log_theta +=noise


    def multNoiseOffDiags(self, amplitude, domain= DOM.TOTAL, col_domain=None):
        if col_domain is None:
            col_domain=domain

        n=self.event_count[self._cStep][DOM.TOTAL]
        mask2d = np.outer(self.getDomainMask(domain), self.getDomainMask(col_domain))   #shape of nxn
        np.fill_diagonal(mask2d, 1)
        noise = np.random.normal(1, amplitude, size=(n,n))* mask2d
        np.fill_diagonal(noise, 1)
        self._mhn[self._cStep].log_theta *=noise


    #this function uses multiplicative noise
    def get_noisy_MHN(self, amplitude, domain= DOM.TOTAL, col_domain=None):
        if col_domain is None:
            col_domain=domain

        n=self.event_count[self._cStep][DOM.TOTAL]
        mask2d = np.outer(self.getDomainMask(domain), self.getDomainMask(col_domain))   #shape of nxn
        np.fill_diagonal(mask2d, 1)
        noise = np.random.normal(1, amplitude, size=(n,n))* mask2d
        np.fill_diagonal(noise, 1)

        mhn_copy= mhn_tools.copyMHN(self.getMHN()) 
        mhn_copy.log_theta *=noise
        
        return mhn_copy

    def getMHN(self)->mhn.model.cMHN:
        return self._mhn[self._cStep]
    
    def getEvents(self, domain:CMG_Domain = DOM.TOTAL)->List[str]:
        return self.events[self._cStep][domain]

    def __str__(self):
        return f"{self.events[self._cStep][DOM.TEST]} | {self.events[self._cStep][DOM.CLUSTER]}\n{self._mhn[self._cStep]}"
     