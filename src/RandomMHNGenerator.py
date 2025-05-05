from enum import Enum
from pathlib import Path
from typing import Dict, List
import mhn
import numpy as np

class RandomMHNGenerator:


    class RMG_Step(Enum):
        SET=0
        TRAINED=1

    global STP
    STP = RMG_Step



    def __init__(self, events:List[str]):
        self._events=events
        self._n = len(events)
        self._mhn : Dict[RandomMHNGenerator.RMG_Step, mhn.model.cMHN]={}
        self._mhn[STP.SET]=mhn.model.cMHN(np.zeros((self._n,self._n)),events=events)
        self._cStep=STP.SET



    def do_training_iteration(self, sample_num:int, penalty:mhn.optimizers.cMHNOptimizer.Penalty = mhn.optimizers.cMHNOptimizer.Penalty.L1):
        print(f"Doing training iteration with {sample_num} sample datapoints.")
        data=self._mhn[self._cStep].sample_artificial_data(sample_num=sample_num)
        opt= mhn.optimizers.cMHNOptimizer()
        opt.load_data_matrix(data)
        opt.set_penalty(penalty)
        lam = opt.lambda_from_cv(show_progressbar=True)
        print(f"lambda from cv: {lam}")
        opt.train(lam=lam)
        self._mhn[STP.TRAINED]= opt._result

        self._cStep=STP.TRAINED
        

    def randomizeBaseRates(self):
        np.fill_diagonal(self._mhn[self._cStep].log_theta,np.random.pareto(3,self._n)-2.2)
        return self
    
    def addNoise(self, amplitude:float):
        self._mhn[self._cStep].log_theta+= np.random.normal(0, amplitude, size=(self._n,self._n))

    def getMHN(self):
        return self._mhn[self._cStep]
    

    def saveto(self, dir:str):
        #save all computed data in directory 'dir'
        Path(dir).mkdir(parents=False, exist_ok=True)
        for step in STP:
            self._mhn[step].save(filename=f"{dir}/mhn_{step.name}")

    def loadfrom(self, dir:str):
        #load all data stored in directory 'dir'
        for step in STP:
            print(f"loading from {dir}/mhn_{step.name}")
            self._mhn[step] = mhn.model.cMHN.load(filename=f"{dir}/mhn_{step.name}", events=self._events)

        self._cStep=STP.TRAINED
