
import sys
import os
import numpy as np
import pickle
import pandas as pd
import pickle
cwd = os.getcwd()
sys.path.append(cwd)

from src.simhash import SH

def create_sh_sk(list_sizes):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    list_total = []
    print(len(vectors))
    for j in list_sizes:
        sketcher = SH(j, j)
        list_sh = []
        for i in vectors:

            sk = sketcher.sketch(i[1])
            list_sh.append((i[0], sk))
        list_total.append(list_sh)
        print("One size done!")

    with open('data/sh_sketches.pkl', 'wb') as f:
        pickle.dump(list_total, f)

def create_sh_sk_faster(list_sizes):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    list_return = []
    print(len(vectors))
    for j in list_sizes:
        sketcher = SH(j, j)
        list_total = []

        list_sh = sketcher.batch_sketch([i[1] for i in vectors])
        for i in range(0, len(list_sh)):
            list_total.append((vectors[i][0], list_sh[i]))

        print("One size done!")
        list_return.append(list_total)

    with open('data/sh_sketches.pkl', 'wb') as f:
        print(list_return)
        pickle.dump(list_return, f)
