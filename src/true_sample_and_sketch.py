import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.optimize import fsolve
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

def angle(v1, v2):
    # Calculate the dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate the magnitudes (L2 norms) of both vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Check for zero vectors to avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Cannot compute the angle with a zero vector.")
        
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Clip cos_theta to the valid range [-1.0, 1.0] 
    # This prevents NaN errors from np.arccos due to floating-point rounding
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians, then convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

class SandSSketch(InnerProdSketch):
    def __init__(self, sampled_vector, sketch, sketch_class, sketch_size, seed, mode='new', vector=None, sketcher=None):
        self.sampled_vector = sampled_vector
        self.sketch = sketch
        self.sketch_class = sketch_class
        self.sketch_size = sketch_size
        self.seed = seed
        self.mode = mode
        self.vector = vector
        self.sketcher = sketcher
    
    def inner_product_asymetric(self, vec, matrix=None):
        sampled_vector = self.sampled_vector.toarray()
        sketcher = self.sketch_class(self.sketch_size, self.seed)
        sketcher.set_matrix(matrix)

        W1 = sampled_vector @ vec
        vec_to_sketch = np.zeros(len(vec))
        vec_to_sketch[sampled_vector == 0] = vec[sampled_vector == 0]
        W2 = self.sketch.inner_product(sketcher.sketch(vec_to_sketch))

        sketcher.set_matrix(None)
        return W1 + W2

    def get_norm(self):
        return np.sqrt(np.linalg.norm(self.sampled_vector) ** 2 + self.sketch.get_norm()**2)

    def inner_product(self, other, matrix=None):
        sketcher = self.sketch_class(self.sketch_size, self.seed)
        idxs = np.where((self.sampled_vector.toarray() != 0) & (other.sampled_vector.toarray() != 0))[0]

        W1, W2, W3, W4 = 0, 0, 0, 0

        self.sketch.set_matrix(matrix)
        other.sketch.set_matrix(matrix)
        W1 = self.sampled_vector.toarray() @ other.sampled_vector.toarray()

        other_sampled_overlap = other.sampled_vector.toarray()
        other_sampled_overlap[idxs] = 0 

        self_sampled_overlap = self.sampled_vector.toarray()
        self_sampled_overlap[idxs] = 0 

        W2 = self.sketch.QJL(other_sampled_overlap)
        W3 = other.sketch.QJL(self_sampled_overlap)
            
        W4 = self.sketch.inner_product(other.sketch)

        self.sketch.set_matrix(None)
        other.sketch.set_matrix(None)

        return W1 + W2 + W3 + W4

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
        sampled_vector = coo_array(sampled_vector)

        vector[idxs] = 0
        # Then I sketch the rest

        sketcher = self.sketch_class(self.sketch_size, self.seed, self.matrix)
        sketch = sketcher.sketch(vector)

        return SandSSketch(sampled_vector, sketch, self.sketch_class, self.sketch_size, self.seed, self.mode, None, None)
    
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
            samples_list.append(coo_array(sampled_vector))

            vector[idxs] = 0
            # Then I sketch the rest

            sh_vecs.append(vector)
            
        sh_list = sketcher.batch_sketch2(sh_vecs)

        for i in range(0, len(vectors)):    
            sh_list[i].set_matrix(self.matrix)
            sketches_list.append(SandSSketch(samples_list[i], sh_list[i], self.sketch_class, self.sketch_size, self.seed, self.mode))

        return sketches_list

    def get_matrix(self):
        return self.matrix
    