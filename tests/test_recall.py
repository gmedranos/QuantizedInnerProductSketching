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
def recall_test_efficient(num_tests, sizesPSQ, sizesSH, num_vectors = None):
    # Bit size = sizesSH
    
    with open('data/arrays_queries.pkl', 'rb') as f:
        vectors_q = pickle.load(f)

    recall_sizes = [50, 100, 500, 1000]

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
    for k in recall_sizes:
        list_answers = []
        for j in range(0, num_tests): 
            q = idx.query(StandardVec(vectors_q[j][1]), k)
            q = set([i[0] for i in q])

            list_answers.append(q)
        answers.append(list_answers)

    del vectors
    print("normal done!")
    # Calculate the output for the PSQ
    
    PSQ_values = []
    for i in range(0, len(sizesSH)):
        with open('data/psq_sketches' + str(sizesSH[i]) + ".pkl", 'rb') as f:
            psq_sketches = pickle.load(f)
        
        psq_sketches = psq_sketches[:num_vectors]
        idxPSQ = VSIndex()
        idxPSQ.insert_vectors(psq_sketches)
        PSQ_answers = []

        for k in recall_sizes:
            sets = []
            ps = PSQ(sizesPSQ[i], sizesPSQ[i] + 2)
            for j in range(0, num_tests): 
                q = idxPSQ.query(ps.sketch_fast(vectors_q[j][1].toarray()), k)
                q = set([i[0] for i in q])

                sets.append(q)
            PSQ_answers.append(sets)
        PSQ_values.append(PSQ_answers)
        del psq_sketches
    
    print("psq done!")
    # Calculate the answers for SH
    SH_values = []
    for i in range(0, len(sizesSH)):
        with open('data/sh_sketches' + str(sizesSH[i]) + ".pkl", 'rb') as f:
            sh_sketches = pickle.load(f)
        
        sh_sketches = sh_sketches[:num_vectors]
        idxSH = VSIndex()
        idxSH.insert_vectors(sh_sketches)
        SH_answers = []

        for k in recall_sizes:
            sets = []
            sh = SH(sizesSH[i], sizesSH[i])
            for j in range(0, num_tests): 
                q = idxSH.query(sh.sketch(vectors_q[j][1]), k)
                q = set([i[0] for i in q])

                sets.append(q)
            SH_answers.append(sets)
        SH_values.append(SH_answers)

        del sh_sketches
    print("sh done!")

    scores_sh = []
    scores_psq = []
    print(PSQ_values[0][0][0])
    for i in range(0, len(sizesSH)):
        scores_recall_sh = []
        scores_recall_psq = []
        for j in range(0, len(recall_sizes)):
            score_sh, score_psq = 0, 0
            for k in range(0, num_tests):
                q = answers[j][k]
                qPS = PSQ_values[i][j][k]
                qSH = SH_values[i][j][k]

                
                score_sh += len(q.intersection(qSH)) / recall_sizes[j]
                score_psq += len(q.intersection(qPS)) / recall_sizes[j]
            
            scores_recall_sh.append(score_sh / num_tests)
            scores_recall_psq.append(score_psq / num_tests)
        scores_sh.append(scores_recall_sh)
        scores_psq.append(scores_recall_psq)
    
    for i in range(0, len(recall_sizes)):
        print("Recall for top " + str(recall_sizes[i]))
        for j in range(0, len(sizesSH)):
            print(scores_sh[j][i], end=' ')
        print()
        for j in range(0, len(sizesSH)):
            print(scores_psq[j][i], end=' ')
        print()

    with open("recall.txt", "w") as text_file:
        for i in range(0, len(recall_sizes)):
            text_file.write("Recall for top " + str(recall_sizes[i]) + '\n')
            for j in range(0, len(sizesSH)):
                text_file.write(str(scores_sh[j][i]) + " ")
            text_file.write('\n')
            for j in range(0, len(sizesSH)):
                text_file.write(str(scores_psq[j][i]) + " ")
            text_file.write('\n')

