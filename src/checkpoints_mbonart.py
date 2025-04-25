import mhn
import numpy as np
import pandas as pd
import os.path


#use this to store the results of EventDistanceMeasurer, RandomMHNGenerator and alike
#these classes should implement the interface CPable (have a method called __hashstr__ which uniquely identifies)



CHECKPOINT_DIR="mbonart_checkpoints/"
FORCE_EXECUTE_COMPUTATIONS=False



def is_already_computed(filename:str)->bool:
    if FORCE_EXECUTE_COMPUTATIONS: return False

    return os.path.isfile(CHECKPOINT_DIR+filename)


def is_dir_already_computed(dirname:str)->bool:
    if FORCE_EXECUTE_COMPUTATIONS: return False

    print(f"check if {CHECKPOINT_DIR+dirname}   does exist")
    return os.path.isdir(CHECKPOINT_DIR+dirname)






