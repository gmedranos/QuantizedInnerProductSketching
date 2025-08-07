import numpy as np
from abc import ABC, abstractmethod
from .abstract_class import InnerProdSketcher, InnerProdSketch, hash_kwise
from scipy.sparse import vstack
import scipy as sp

class SHSketch(InnerProdSketch):
    def __init__(self, norm, signs, sketch_size):
        self.norm = norm
        self.signs = signs
        self.sketch_size = sketch_size

    def inner_product(self, other):
        sum = np.sum(other.signs == self.signs)

        return self.norm * other.norm * np.cos((1 - sum / self.sketch_size) * np.pi)

class SH(InnerProdSketcher):
    def __init__(self, sketch_size, seed):
        self.sketch_size = sketch_size
        self.seed = seed

    def sketch(self, vector):
        dim = vector.shape[0]
        vec = []
        rng = np.random.default_rng(seed = self.seed)
        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)

            vec.append(vector.dot(R))
        
        vec = np.array(vec)

        return SHSketch(np.linalg.norm(vector.toarray()), np.sign(vec), self.sketch_size)
    
    def sketch_all(self, vectors, batch_size = 50000):
        list_of_sk = []

        for i in range(0, len(vectors) // batch_size + 1):
            list_of_sk = list_of_sk + self.batch_sketch(vectors[i*batch_size : (i + 1) *batch_size + 1])
        return list_of_sk

    # Get a list of vectors to sketch
    def batch_sketch(self, vectors):
        vectors = [v.reshape(1, -1) for v in vectors]
        dim = vectors[0].shape[1]
        rng = np.random.default_rng(seed=self.seed)
        num_vectors = len(vectors)
        vector_matrix = vstack(vectors)
        sh_matrix = []

        for i in range(0, self.sketch_size):
            R = rng.normal(0, 1, dim)
            sh = vector_matrix @ R
            sh_matrix.append(np.sign(sh))
        
        lists = list(np.array(sh_matrix).T)

        list_of_sk = []
        count = 0
        for i in lists:
            list_of_sk.append(SHSketch(np.linalg.norm(vectors[count].toarray()), np.array(i, dtype=np.int8), self.sketch_size))
            count += 1
        
        return list_of_sk
    