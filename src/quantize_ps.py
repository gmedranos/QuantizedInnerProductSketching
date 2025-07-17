import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve

import hashlib

def seeded_hash(value, seed):
    seed_bytes = str(seed).encode()
    value_bytes = value.encode()
    hash = hashlib.blake2b(value_bytes, digest_size=8, key = seed_bytes).digest()

    return int.from_bytes(hash, byteorder='big')

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

def eq_epsilon(epsilon, n, intervals):
    return  np.log2(n / epsilon)  / np.log2(1 + epsilon) + 2 - intervals

def find_epsilon(b, n):
    return fsolve(lambda x: eq_epsilon(x, n, 2**b), 0.01)[0]

def myhash(a, seed):
    h = seeded_hash(str(a), str(seed))
    return h / 2**64

def hash1(a, seed):
    h = seeded_hash(str(a), (seed))
    return (h // 2**63 - 0.5) * 2 // 1

def hash32int(a, seed, bits):
    h = seeded_hash(str(a), str(seed))
    return h // 2**(64 - bits)


class PSQSketch(InnerProdSketch):
    def __init__(self, K, t, norm, seed):
        self.K = K
        self.t = t
        self.norm = norm
        self.collisions = 0
        self.seed = seed

    def list_prod(self, listA, listB, ta, tb):
        soma = 0
        for i in listA:
            for j in listB:
                soma += i[0]*j[0]*i[1]*j[1] / min(1, i[0] ** 2 * ta, j[0] ** 2 * tb)
        return soma

    def inner_product(self, other):
        idxA = set(self.K.keys())
        idxB = set(other.K.keys())

        inter = idxA.intersection(idxB)
        sum = 0 
        for i in inter:
            sum += self.list_prod(self.K[i], other.K[i], self.t, other.t)
        return sum * self.norm * other.norm
    

class PSQ(InnerProdSketcher):
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
    
    def sketch(self, vector : np.ndarray) -> PSQSketch:
        dim = len(vector)
        idx = np.arange(dim)
        # Instead of Ka and Va, I'll just use a dictionary from idx to value
        Ka = {}
        norm = np.linalg.norm(vector)

        vector = vector / norm
        vector = quantize(find_epsilon(4, len(vector)), vector)
        #vector = quantize(0.00003, vector)

        vHash = np.vectorize(myhash)
        vHash.excluded.add(1)

        idx_hash = vHash(idx, self.seed)
        Ra = idx_hash / vector**2


        if (self.sketch_size < dim):
            ta = np.sort(Ra)[self.sketch_size]
        else:
            ta = np.inf
        
        # Slow
        for i in range(0, dim):
            if(Ra[i] < ta):
                idx_h = hash32int(i, self.seed + 2, 12)
                #idx_h = i
                if(idx_h not in Ka):
                    Ka[idx_h] = [(vector[i], hash1(i, self.seed + 1))]
                else:
                    Ka[idx_h].append((vector[i], hash1(i, self.seed + 1)))

        return PSQSketch(Ka, ta, norm, self.seed)