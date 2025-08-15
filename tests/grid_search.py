
from tests.test_recall import recall_test_efficient
from utils.create_PSQ import create_PS_separete
import pickle

def grid_search(bit_sizes, bits, recall, num_vectors):
    results = {}
    for i in bits:
        sizes = [j // (i[0] + i[1] + 1) for j in bit_sizes]
        create_PS_separete(sizes, bit_sizes, i[1], i[0], path='data/psq_grid', num_vectors=num_vectors)
        sh, ps = recall_test_efficient(30, sizes, bit_sizes, recall_sizes=recall, num_vectors=num_vectors, path='data/psq_grid')
        results[i] = (sh, ps)
    
    with open('data/grid_search.pkl', 'wb') as f:
        pickle.dump(results, f)