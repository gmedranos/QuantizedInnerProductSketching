
import sys
import os
import numpy as np
import pickle
import pandas as pd
import pickle
cwd = os.getcwd()
sys.path.append(cwd)

from src.true_sample_and_sketch import Sample_and_sketch
from src.simhash import SH

def create_sh_sk(list_sizes):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    list_total = []
    print(len(vectors))
    for j in list_sizes:
        sketcher = Sample_and_sketch(int(0.6 * j), SH, int(0.4 * j), j, dim = len(vectors[0]))
        list_sh = []
        for i in vectors:

            sk = sketcher.sketch(i[1])
            list_sh.append((i[0], sk))
        list_total.append(list_sh)
        print("One size done!")

    with open('data/sas_sketches.pkl', 'wb') as f:
        pickle.dump(list_total, f)


def create_sas_separate(list_sizes, bit_sizes= [256, 512, 1024, 2048], num_vectors=None, path='data/sas_sketches'):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)


    if num_vectors is not None:        
        vectors = vectors[:num_vectors]
    
    
    list_total = []
    count = 0
    for j in list_sizes:
        print(j)
        sketcher = Sample_and_sketch(int(0.6 * j), SH, int(0.4 * j), j, dim = vectors[0][1].shape[0])
        list_sh = []
        for i in vectors:
            # Turn into a dense representation
            sk = sketcher.sketch(i[1].toarray())
            list_sh.append((i[0], sk))

        
        with open(path + str(bit_sizes[count]) + '.pkl', 'wb') as f:
            pickle.dump(list_sh, f)
        count += 1