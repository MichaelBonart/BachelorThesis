import mhn
import numpy as np
import pandas as pd
import sys
import os.path

from os.path import dirname, abspath
from pathlib import Path

#use this to store the results of EventDistanceMeasurer, RandomMHNGenerator and alike
#these classes should implement the interface CPable (have a method called __hashstr__ which uniquely identifies)



CHECKPOINT_DIR=Path(dirname(abspath(__file__))) / "mbonart_checkpoints"
FORCE_EXECUTE_COMPUTATIONS=False
print(f"Checkpoint directory is: {CHECKPOINT_DIR}")


def is_already_computed(filename:str)->bool:
    if FORCE_EXECUTE_COMPUTATIONS: return False

    return os.path.isfile(CHECKPOINT_DIR/filename)


def is_dir_already_computed(dirname:str)->bool:
    if FORCE_EXECUTE_COMPUTATIONS: return False

    print(f"check if {CHECKPOINT_DIR/dirname}   does exist")
    return os.path.isdir(CHECKPOINT_DIR/dirname)






