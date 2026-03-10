
import sys
import os
import numpy as np
import pickle
import pandas as pd
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from src.AMS_quantize import AMSQ


def create_AMSQ_separete(list_sizes, bit_sizes= [256, 512, 1024, 2048], key_size=12, float_size=4, num_vectors=None, path='data/amsq_sketches'):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    if num_vectors is not None:        
        vectors = vectors[:num_vectors]
    
    list_total = []
    count = 0
    for j in list_sizes:
        print(j)
        sketcher = AMSQ(j, j, float_size=float_size, key_size=key_size)
        list_sh = []
        for i in vectors:
            # Turn into a dense representation
            sk = sketcher.sketch(i[1].toarray())
            list_sh.append((i[0], sk))

        
        with open(path + str(bit_sizes[count]) + '.pkl', 'wb') as f:
            pickle.dump(list_sh, f)
        count += 1


