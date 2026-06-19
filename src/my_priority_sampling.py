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


class PSSketch(InnerProdSketch):
    def __init__(self, K, t, norm, seed, size):
        self.K = K
        self.t = t
        self.norm = norm
        self.collisions = 0
        self.seed = seed
        self.size = size


    def get_vector(self):
        vec = np.zeros(self.size)
        vec[self.K[0]] = self.K[1]
        vec_weighted = np.where(vec == 0, 0, vec / np.minimum(1, vec ** 2 * self.t))
        #print(np.minimum(1, vec ** 2 * self.t))
        return vec_weighted

    def inner_product(self, other, matrix):
        set1 = set(self.K[0])
        set2 = set(other.K[0])

        W1 = 0
        for i in set1.intersection(set2):
            W1 += np.float64(self.K[1][self.K[0] == i]) * np.float32(other.K[1][other.K[0] == i]) / min(1, np.float32(self.K[1][self.K[0] == i]) ** 2 * self.t, np.float32(other.K[1][other.K[0] == i]) ** 2 * other.t)

        return W1

class PS(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector : np.ndarray) -> PSSketch:
        dim = len(vector)

        norm = np.linalg.norm(vector)
        idxs = np.arange(len(vector))

        idx_hash = hash_kwise(vector, self.seed)[0]
        Ra = idx_hash / vector**2

        number_of_samples = min(dim, self.sketch_size // int(64 + 32))
        partition = np.argsort(Ra)
        
        ta = Ra[partition[number_of_samples]]
   
        idxs = partition[:number_of_samples].copy()

        values = np.float64(vector[idxs].copy())
  
        return PSSketch((idxs, values), ta, norm, self.seed, dim)
    
    def get_matrix(self):
        return