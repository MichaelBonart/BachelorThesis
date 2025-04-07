import mhn
import numpy as np
import pandas as pd
import os.path


CHECKPOINT_DIR="mbonart_checkpoints/"
FORCE_EXECUTE_COMPUTATIONS=False



def is_already_computed(filename:str)->bool:
    if FORCE_EXECUTE_COMPUTATIONS: return False

    return os.path.isfile(CHECKPOINT_DIR+filename)
