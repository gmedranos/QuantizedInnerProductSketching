import sys
import os
import numpy as np
import pickle
import random
from src.simhash import SH
from src.quantize_ps import PSQ
from pympler import asizeof

import gc

class StandardVec():
    def __init__(self, vec):
        self.vec = vec
    def inner_product(self, other):
        return other.vec.toarray().dot(self.vec.toarray())

cwd = os.getcwd()
sys.path.append(cwd)
from src.vsindex import VSIndex


def recall_test(num_tests, sizesPSQ, sizesSH):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)
        
    with open('data/arrays_queries.pkl', 'rb') as f:
        vectors_q = pickle.load(f)

    with open('data/psq_sketches.pkl', 'rb') as f:
        vectors_sh = pickle.load(f)

    with open('data/psq_sketches.pkl', 'rb') as f:
        vectors_psq = pickle.load(f)

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

            idx.insert_vectors([(j[0], StandardVec(j[1])) for j in vectors])
            idxPS.insert_vectors(vectors_psq[i])
            idxSH.insert_vectors(vectors_sh[i])

            sh = SH(sizesSH[i], sizesSH[i])
            ps = PSQ(sizesPSQ[i], sizesPSQ[i] + 2)
            for j in range(0, num_tests): 
                #print(np.size(np.nonzero(vectors_q[j][1])))
                s = ps.sketch_fast(vectors_q[j][1].toarray())
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

# This is more memory efficient
def recall_test_efficient(num_tests, sizesPSQ, sizesSH, num_vectors = None, recall_sizes=[50, 100, 500, 1000], path='data/psq_sketches'):
    # Bit size = sizesSH
    
    with open('data/arrays_queries.pkl', 'rb') as f:
        vectors_q = pickle.load(f)

    # Calculate the answers
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    if(num_vectors is None):
        num_vectors = len(vectors)

    vectors = vectors[:num_vectors]
    stand_list = [(j[0], StandardVec(j[1])) for j in vectors]
    idx = VSIndex()
    idx.insert_vectors(stand_list)
    
    answers = []

    for j in range(0, num_tests): 
        q = idx.query(StandardVec(vectors_q[j][1]), recall_sizes[-1])
        
        list_answers = []
        best = q[0]
        for k in recall_sizes:
            
            list_answers.append(set([i[0] for i in q[:k]]))
        answers.append(list_answers)
    
    del vectors
    print("normal done!")
    # Calculate the output for the PSQ
    
    PSQ_values = []
    for i in range(0, len(sizesSH)):
        size_answers = []
        with open(path + str(sizesSH[i]) + ".pkl", 'rb') as f:
            psq_sketches = pickle.load(f)
        
        psq_sketches = psq_sketches[:num_vectors]
        idxPSQ = VSIndex()
        idxPSQ.insert_vectors(psq_sketches)

        # I probably only need to this once, for the bigger recall size, such that for the smaller size I just take the slice of the bigger one
        ps = PSQ(sizesPSQ[i], sizesPSQ[i] + 2)


        for j in range(0, num_tests): 
            q = idxPSQ.query(ps.sketch_fast(vectors_q[j][1].toarray()), recall_sizes[-1])
            
            PSQ_answers = []
            '''
            print("Query:")
            print(ps.sketch_fast(vectors_q[j][1].toarray()).K)
            print(ps.sketch_fast(vectors_q[j][1].toarray()).t)

            print("PSQ best:")
            print(q[0][1].t)
            print(q[0][1].K)

            print("Real best:")
            print(ps.sketch_fast(best[1].vec.toarray()).t)
            print(ps.sketch_fast(best[1].vec.toarray()).K)
            '''
            for k in recall_sizes:
                test_answer = set([i[0] for i in q[:k]])
                PSQ_answers.append(test_answer)
            size_answers.append(PSQ_answers)
        
        PSQ_values.append(size_answers)
        del psq_sketches
    
    print("psq done!")
    # Calculate the answers for SH
    SH_values = []
    for i in range(0, len(sizesSH)):
        size_answers = []
        with open('data/sh_sketches' + str(sizesSH[i]) + ".pkl", 'rb') as f:
            sh_sketches = pickle.load(f)
        
        sh_sketches = sh_sketches[:num_vectors]
        idxSH = VSIndex()
        idxSH.insert_vectors(sh_sketches)

        sh = SH(sizesSH[i], sizesSH[i])

        for j in range(0, num_tests): 
            q = idxSH.query(sh.sketch(vectors_q[j][1]), recall_sizes[-1])
            SH_answers = []

            for k in recall_sizes:
                test_answer = set([i[0] for i in q[:k]])
                SH_answers.append(test_answer)
            size_answers.append(SH_answers)
        
        SH_values.append(size_answers)
        del sh_sketches
    print("sh done!")

    scores_sh = []
    scores_psq = []
    for i in range(0, len(sizesSH)):
        score_sh, score_psq = np.zeros(len(recall_sizes)), np.zeros(len(recall_sizes))

        for j in range(0, num_tests):
            for k in range(0, len(recall_sizes)):
                q = answers[j][k]
                qPS = PSQ_values[i][j][k]
                qSH = SH_values[i][j][k]

                score_sh[k] += len(q.intersection(qSH)) / recall_sizes[k]
                score_psq[k] += len(q.intersection(qPS)) / recall_sizes[k]
               
        scores_sh.append(score_sh / num_tests)
        scores_psq.append(score_psq / num_tests)
    
    for i in range(0, len(recall_sizes)):
        print("For k = " + str(recall_sizes[i]))
        print(np.array(scores_sh).T[i])
        print(np.array(scores_psq).T[i])

    return np.array(scores_sh).T, np.array(scores_psq).T


