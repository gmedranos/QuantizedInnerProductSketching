import sys
import os
import numpy as np
import pickle
cwd = os.getcwd()
sys.path.append(cwd)
from src.simhash import SH
from src.quantize_ps import PSQ
import matplotlib.pyplot as plt


def test_ip(sizesSH, sizesPSQ, trials):
    with open('data/arrays.pkl', 'rb') as f:
        vectors = pickle.load(f)

    with open('data/arrays_queries.pkl', 'rb') as f:
        vectors_q = pickle.load(f)

    errors_sh = []
    errors_psq = []


    for j in range(len(sizesSH)):
        error_psq = 0
        error_sh = 0
        aaaa = 0
        for i in range(0, trials):

            #aaa = np.random.randint(0, 999)
            #bbb = np.random.randint(0, 999)
            seed = np.random.randint(10000)
            vec = vectors_q[i][1]
            k = np.random.randint(len(vectors))

            a = set(np.nonzero(vec)[0])
            vec_2 = vectors[k][1]
            b = set(np.nonzero(vec_2)[0])


            idx = np.nonzero((vec != 0) & (vec_2 != 0))[0]
            our = (max(np.linalg.norm(vec[idx]) * np.linalg.norm(vec_2), np.linalg.norm(vec_2[idx]) * np.linalg.norm(vec)))
            their = (np.linalg.norm(vec)*np.linalg.norm(vec_2))

            aaaa += our / their

            sh = SH(sizesSH[j], seed)
            psq = PSQ(sizesPSQ[j], seed)

            ip = vec @ vec_2

            vec_sh = sh.sketch(vec)
            vec_sh2 = sh.sketch(vec_2)

            vec_psq = psq.sketch(vec)
            vec_psq2 = psq.sketch(vec_2)


            v_sh = vec_sh2.inner_product(vec_sh)
            v_psq = vec_psq2.inner_product(vec_psq)

            
            error_sh += np.abs(ip - v_sh) / (np.linalg.norm(vec) * np.linalg.norm(vec_2))
            error_psq += np.abs(ip - v_psq) / (np.linalg.norm(vec) * np.linalg.norm(vec_2))
        
        errors_psq.append(error_psq / trials)
        errors_sh.append(error_sh / trials)
        print("?")
        print(aaaa / trials)
            
    print(errors_sh)
    print(errors_psq)
    print(sizesSH)

    plt.plot(sizesSH, errors_sh, marker='^', label='SH', color='green')
    plt.plot(sizesSH, errors_psq, marker='x', label='PSQ', color='red')


    plt.xlabel("Storage Size in bits")
    plt.ylabel("Scaled Average Difference")
    plt.title("Comparison of Methods")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig('figs/ip_test.png')