import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve

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


def quantize(epsilon, a):
    dim = len(a)
    new_a = np.zeros(dim)
    sinal = np.sign(a)
    a = a* sinal
    for i in range(0, dim):
        # Calculate to which bin the value belongs
        idx = np.round(np.log(a[i]*dim / epsilon) / np.log(1 + epsilon))
        # Round the number
        if(a[i] >= epsilon / dim):
            new_a[i] = min(epsilon*(1+epsilon)**idx / dim, 1)
        else:
            new_a[i] = 0
    return new_a*sinal

def quantize_faster(epsilon, a):
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


class SandSSketch(InnerProdSketch):
    def __init__(self, sampled_vector, sketch, sketch_class, sketch_size, seed, mode='new'):
        self.sampled_vector = sampled_vector
        self.sketch = sketch
        self.sketch_class = sketch_class
        self.sketch_size = sketch_size
        self.seed = seed
        self.mode = mode

    def inner_product(self, other):
        sketcher = self.sketch_class(self.sketch_size, self.seed)

        W1, W2, W3, W4 = 0, 0, 0, 0

        W1 = self.sampled_vector @ other.sampled_vector
        if(self.mode != 'new'):
            W2 = sketcher.sketch(self.sampled_vector).inner_product(other.sketch)
            W3 = sketcher.sketch(other.sampled_vector).inner_product(self.sketch)
        else:
            W2 = self.sketch.QJL(other.sampled_vector)
            W3 = other.sketch.QJL(self.sampled_vector)
            
        W4 = self.sketch.inner_product(other.sketch)

        #print(self.sampled_vector @ other.sampled_vector, " ", self.vector @ other.sampled_vector, " ", self.sampled_vector @ other.vector, " ", self.vector @ other.vector)
        #print(W1, " ", W2, " ", W3, " ", W4)
        return W1 + W2 + W3 + W4
    
    def get_norm(self):
        return np.sqrt(np.linalg.norm(self.sampled_vector) ** 2 + self.sketch.get_norm()**2)

class Sample_and_sketch(InnerProdSketcher):
    def __init__(self, sketch_size: int, sketch_class, sample_size: int, seed: int, mode='new', dim=None) -> None:
        self.sample_size: int = sample_size
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.sketch_class = sketch_class
        self.mode = mode
        self.matrix = None

        if(dim != None):
            # Get the random matrix to make QJL faster
            rng = np.random.default_rng(seed = self.seed)
            self.matrix = rng.normal(0, 1, (sketch_size, dim))



    def sketch(self, vector : np.ndarray) -> SandSSketch:
        dim = len(vector)
        vector = vector.copy()

        # First I sample items
        norm = np.linalg.norm(vector)
        idxs = np.arange(len(vector))

        idx_hash = hash_kwise(vector, self.seed)[0]
        Ra = idx_hash / vector**2

        number_of_samples = min(dim, self.sample_size // int(np.ceil(np.log2(dim)) + 16))
        partition = np.argsort(Ra)
        
        ta = Ra[partition[number_of_samples]]
   
        idxs = partition[:number_of_samples].copy()
        values = np.float16(vector[idxs].copy())

        sampled_vector = np.zeros(dim)
        sampled_vector[idxs] = values

        vector[idxs] = 0
        # Then I sketch the rest

        sketcher = self.sketch_class(self.sketch_size, self.seed, self.matrix)
        sketch = sketcher.sketch(vector)
        sketch.set_matrix(self.matrix)

        return SandSSketch(sampled_vector, sketch, self.sketch_class, self.sketch_size, self.seed, self.mode)
    
    def batch_sketch(self, vectors):
        sketches_list = []
        samples_list = []
        sh_list = []
        sh_vecs = []
        dim = len(vectors[0])
        sketcher = self.sketch_class(self.sketch_size, self.seed, self.matrix)

        for vector in vectors:
            # First I sample items
            norm = np.linalg.norm(vector)
            idxs = np.arange(len(vector))

            idx_hash = hash_kwise(vector, self.seed)[0]
            Ra = idx_hash / vector**2

            number_of_samples = min(dim, self.sample_size // int(np.ceil(np.log2(dim)) + 16))
            partition = np.argsort(Ra)
            
            ta = Ra[partition[number_of_samples]]
    
            idxs = partition[:number_of_samples].copy()
            values = np.float16(vector[idxs].copy())

            sampled_vector = np.zeros(dim)
            sampled_vector[idxs] = values
            samples_list.append(sampled_vector)

            vector[idxs] = 0
            # Then I sketch the rest

            sh_vecs.append(vector)
            
        sh_list = sketcher.batch_sketch2(sh_vecs)

        for i in range(0, len(vectors)):    
            sh_list[i].set_matrix(self.matrix)
            sketches_list.append(SandSSketch(samples_list[i], sh_list[i], self.sketch_class, self.sketch_size, self.seed, self.mode))

        return sketches_list
    