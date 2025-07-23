
import sys
import os
import numpy as np
import pickle
import pandas as pd
import pickle
cwd = os.getcwd()
sys.path.append(cwd)

from src.quantize_ps import PSQ

def create_sk_PS(list_sizes):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    list_total = []
    sks = []

    for j in list_sizes:
        print(j + 2)
        sketcher = PSQ(j, j + 2)
        list_sh = []
        for i in vectors:
            # Turn into a dense representation
            sk = sketcher.sketch(i[1].toarray()[0])
            list_sh.append((i[0], sk))
        list_total.append(list_sh)
        print("One size done!")
        sks.append(sketcher)

    with open('data/psq_sketches.pkl', 'wb') as f:
        pickle.dump(list_total, f)
    with open('data/psq_sketchers.pkl', 'wb') as f:
        pickle.dump(sks, f)

