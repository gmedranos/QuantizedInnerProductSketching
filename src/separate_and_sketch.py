import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve
from .priority_sampling import PSSketch, PS
from scipy.sparse import coo_array

from collections import defaultdict


import hashlib

def seeded_hash(value, seed):
    seed_bytes = seed.encode()
    value_bytes = value.encode()
    hash = hashlib.blake2b(value_bytes, digest_size=8, key = seed_bytes).digest()

    return int.from_bytes(hash, byteorder='big')

def seeded_hash1(value, seed):
    value = value + seed
    value ^= value >> 16
    value *= 0x21f0aaad
    value ^= value >> 15
    value *= 0x735a2d97
    value ^= value >> 15
    return value & 0xFFFFFFFF


def quantize_faster(bits, a):
    epsilon = find_epsilon(bits, len(a))
    dim = len(a)
    new_a = np.zeros(dim)
    sinal = np.sign(a)
    a = a* sinal

    idx = np.round(np.log(a *dim / epsilon) / np.log(1 + epsilon))
    new_a = epsilon*(1 + epsilon) ** idx / dim
    new_a = np.minimum(new_a, 1)
    new_a[new_a < epsilon / dim] = 0
    
    return new_a*sinal


def eq_epsilon(epsilon, n, intervals):
    return  np.log2(n / epsilon)  / np.log2(1 + epsilon) + 2 - intervals

def find_epsilon(b, n):
    return fsolve(lambda x: eq_epsilon(x, n, 2**b), 0.01)[0]

def myhash(a, seed):
    h = seeded_hash(str(a), str(seed))
    return h / 2**64

def hash1(a, seed):
    h = seeded_hash(str(a), str(seed))
    return (h // 2**63 - 0.5) * 2 // 1

def hash32int(a, seed, bits):
    h = seeded_hash(str(a), str(seed))
    return h // 2**(64 - bits)

def myhash1(a, seed):
    h = seeded_hash1(a, seed)
    return h / 2**32

def hash11(a, seed):
    h = seeded_hash1(a, seed)
    return (h // 2**31 - 0.5) * 2 // 1

def hash321int(a, seed, bits):
    h = seeded_hash1(a, seed)
    return h // 2**(32 - bits)

import numpy as np

def compute_threshold(a, x):
    n = len(a)
    vec_sorted = sorted(a)
    return vec_sorted[min(n - x - 1, n - 1)]

def compute_var(ah, at, bh, bt, sketch):
    norm_ah = np.linalg.norm(ah)**2
    norm_at = np.linalg.norm(at)**2
    norm_bh = np.linalg.norm(bh)**2
    norm_bt = np.linalg.norm(bt)**2
    return (norm_ah * norm_bt + norm_at * norm_bh + norm_at * norm_bt) / sketch


class SASSketch(InnerProdSketch):
    def __init__(self, head_vec, tail_sketch, seed, sketcher_class_tail, tail_size, head_norm, true_tail=None):
        self.head_vec = head_vec
        self.tail_sketch = tail_sketch
        self.sketcher_class_tail = sketcher_class_tail
        self.tail_size = tail_size
        self.seed = seed
        self.head_norm = head_norm
        self.true_tail = true_tail
    
    def inner_product(self, other, matrix=None):
        sketcher = self.sketcher_class_tail(self.tail_size, self.seed)
                
        head_vec = self.head_vec.astype(np.float32).toarray()
        other_head_vec = other.head_vec.astype(np.float32).toarray()
        idxs = np.where((head_vec != 0) & (other_head_vec != 0))[0]


        W1, W2, W3, W4 = 0, 0, 0, 0

        W1 = np.float64(head_vec) @ np.float64(other_head_vec)

        self.tail_sketch.set_matrix(matrix)
        other.tail_sketch.set_matrix(matrix)

        other_sampled_overlap = np.copy(other_head_vec)
        other_sampled_overlap[idxs] = 0 

        self_sampled_overlap = np.copy(head_vec)
        self_sampled_overlap[idxs] = 0 

        W2 = self.tail_sketch.inner_product_asym(other_sampled_overlap)
        W3 = other.tail_sketch.inner_product_asym(self_sampled_overlap)

        self.tail_sketch.set_matrix(None)
        other.tail_sketch.set_matrix(None)
            
        W4 = self.tail_sketch.inner_product(other.tail_sketch)

        #print(compute_var(self.head_vec, self.true_tail, other.head_vec, other.true_tail, self.tail_size))
        #print(W1, " ", W2, " ", W3, " ", W4)
        #print(self.head_vec @ other.head_vec, " ", self.true_tail @ other.head_vec, " ", other.true_tail @ self.head_vec, " ", self.true_tail @ other.true_tail)

        return W1 + W2 + W3 + W4
    
    '''    def inner_product_asym(self, vec, matrix=None):
        head_vec = self.head_vec.astype(np.float32).toarray()
        self.tail_sketch.set_matrix(matrix)

        W1 = head_vec @ vec
        vec_to_sketch = np.zeros(len(vec))
        vec_to_sketch[head_vec == 0] = vec[head_vec == 0]
        W2 = self.tail_sketch.inner_product_asym(vec_to_sketch)

        self.tail_sketch.set_matrix(None)
        return W1 + W2
    '''
    def inner_product_asym(self, vec, matrix=None):
        head_vec = self.head_vec.astype(np.float32).toarray()
        sketcher = self.sketcher_class_tail(self.tail_size, self.seed)
        sketcher.set_matrix(matrix)

        W1 = head_vec @ vec
        vec_to_sketch = np.zeros(len(vec))
        vec_to_sketch[head_vec == 0] = vec[head_vec == 0]
        W2 = self.tail_sketch.inner_product(sketcher.sketch(vec_to_sketch))

        sketcher.set_matrix(None)
        return W1 + W2

class SaS(InnerProdSketcher):
    def __init__(self, head_size, tail_size, seed: int, tail_sketch_class, dim=None) -> None:
        self.head_size = head_size
        self.tail_size = tail_size
        self.seed: int = seed
        self.tail_sketch_class = tail_sketch_class
        self.matrix = None
        if(dim != None):
            # Get the random matrix to make QJL faster
            rng = np.random.default_rng(seed = self.seed)
            self.matrix = rng.normal(0, 1, (tail_size, dim))
        

    def sketch(self, vector : np.ndarray):
        dim = len(vector)
        norm = np.linalg.norm(vector)
        
        T = compute_threshold(np.abs(vector), self.head_size // int (np.ceil(np.log2(dim)) + 16))

        idxs = np.arange(len(vector))

        idxs_big = idxs[np.abs(vector) > T]
        idxs_small = idxs[np.abs(vector) <= T]

        #print(idxs_big)

        vector_head = np.zeros_like(vector)
        vector_tail = np.zeros_like(vector)

        vector_head[idxs_big] = vector[idxs_big]
        head_norm = np.linalg.norm(vector_head)

        vector_head = np.float16(vector_head)
        vector_tail[idxs_small] = vector[idxs_small]

        sketcher_small = self.tail_sketch_class(self.tail_size, self.seed, self.matrix)
        sketch_small = sketcher_small.sketch(vector_tail)
        #print(np.linalg.norm(vector_head), np.linalg.norm(vector_tail))
        #print(compute_var(vector_head, vector_tail, vector_head, vector_tail, self.tail_size))


        return SASSketch(coo_array(vector_head), sketch_small, self.seed, self.tail_sketch_class, self.tail_size, head_norm, vector_tail)
    
    def sketch_asym(self, vector : np.ndarray):
        return self.sketch(vector)
    
    def get_matrix(self):
        return self.matrix
