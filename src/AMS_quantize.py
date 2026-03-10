import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve
from .priority_sampling import PSSketch, PS

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


class AMSQSketch(InnerProdSketch):
    def __init__(self, K, t, norm, seed):
        self.K = K
        self.norm = norm
        self.seed = seed
        self.t = t

    def inner_product(self, other):
        set1 = set(self.K[0])
        set2 = set(other.K[0])

        sum = 0 
        for i in set1.intersection(set2):
            sum += self.K[1][self.K[0] == i][0] * other.K[1][other.K[0] == i][0] / min(1, self.t * self.K[1][self.K[0] == i][0] ** 2, other.t * other.K[1][other.K[0] == i][0] ** 2)

        return self.norm * other.norm * sum

class AMSQ(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int, float_size=4, key_size= 12) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.float_size = float_size
        self.key_size = key_size

    def sketch(self, vector : np.ndarray):
        dim = len(vector)
        # Instead of Ka and Va, I'll just use a dictionary from idx to value

        vHash = np.vectorize(hash1)
        vHash.excluded.add(1)

        vHash2 = np.vectorize(hash32int)
        vHash2.excluded.add(1)
        vHash2.excluded.add(2)
        
        idxs = np.arange(len(vector))
        small_vec = np.zeros(2 ** self.key_size)
        hashed_keys = vHash2(idxs[vector != 0], self.seed + 2, self.key_size)

        np.add.at(small_vec, hashed_keys, vector[vector != 0] * vHash(idxs[vector != 0], self.seed + 1))

        norm = np.linalg.norm(small_vec)
        small_vec = small_vec / norm
        small_vec = quantize_faster(find_epsilon(self.float_size, len(small_vec)), small_vec)

        idx_hash = hash_kwise(small_vec, self.seed)[0]
        Ra = idx_hash / small_vec**2

        self.sketch_size = min(2**self.key_size - 1, self.sketch_size)
        partition = np.argsort(Ra)
        ta = Ra[partition[self.sketch_size]]
            
        idxs = partition[:self.sketch_size].copy()
        idxs = np.int16(idxs)
        values = small_vec[idxs].copy()

        return AMSQSketch((idxs, values), ta, norm, self.seed)
