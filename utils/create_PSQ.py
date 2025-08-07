
import sys
import os
import numpy as np
import pickle
import pandas as pd
import sys

import gc
cwd = os.getcwd()
sys.path.append(cwd)

from src.quantize_ps import PSQ

def create_sk_PS(list_sizes):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    list_total = []

    for j in list_sizes:
        print(j + 2)
        sketcher = PSQ(j, j + 2)
        list_sh = []
        for i in vectors:
            # Turn into a dense representation
            sk = sketcher.sketch_fast(i[1].toarray())
            list_sh.append((i[0], sk))
        list_total.append(list_sh)

    with open('data/psq_sketches.pkl', 'wb') as f:
        pickle.dump(list_total, f)


