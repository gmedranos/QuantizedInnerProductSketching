import sys
import os
import numpy as np
import pickle
import random
from src.simhash import SH
from src.quantize_ps import PSQ

class StandardVec():
    def __init__(self, vec):
        self.vec = vec
    def inner_product(self, other):
        return self.vec @ other.vec

cwd = os.getcwd()
sys.path.append(cwd)
from src.vsindex import VSIndex


def recall_test(num_tests, sizesPSQ, sizesSH):
    with open('data/sh_sketches.pkl', 'rb') as f:
        vectors_sh = pickle.load(f)

    with open('data/psq_sketches.pkl', 'rb') as f:
        vectors_psq = pickle.load(f)

    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    with open('data/arrays_queries.pkl', 'rb') as f:
        vectors_q = pickle.load(f)

    with open('data/psq_sketchers.pkl', 'rb') as f:
        sks = pickle.load(f)

    recall_sizes = [50, 100, 500, 1000]
    
    for k in recall_sizes:
        recallPS = []
        recallSH = []
        for i in range(0, len(sizesPSQ)):
            idxPS = VSIndex()
            idxSH = VSIndex()
            idx = VSIndex()
            
            rec_sh = 0
            rec_ps = 0

            idx.insert_vectors([(i[0], StandardVec(i[1])) for i in vectors])
            idxPS.insert_vectors(vectors_psq[i])
            idxSH.insert_vectors(vectors_sh[i])

            sh = SH(sizesSH[i], sizesSH[i])
            ps = sks[i]
            for j in range(0, num_tests): 
                #print(np.size(np.nonzero(vectors_q[j][1])))
                s = ps.sketch(vectors_q[j][1])
                ss = sh.sketch(vectors_q[j][1])    

                q = idx.query(StandardVec(vectors_q[j][1]), k)
                qPS = idxPS.query(s, k)
                qSH = idxSH.query(ss, k)



                qPS = set([i[0] for i in qPS])
                q = set([i[0] for i in q])
                qSH = set([i[0] for i in qSH])

                rec_ps += len(q.intersection(qPS)) / k
                rec_sh += len(q.intersection(qSH)) / k

            recallPS.append(rec_ps / num_tests)
            recallSH.append(rec_sh / num_tests)    
        print("For k = "  + str(k) + ":")
        print(recallSH)
        print(recallPS)





